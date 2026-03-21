"""
Explicit time-window averages for :class:`~pandas.DatetimeIndex` frames (LR0100-style).

This is **not** :meth:`pandas.DataFrame.resample` semantics. For **floor / ceiling / center**
windows, monthly label trimming, coverage rules, and examples, see
``docs/tutorials/3.time_averaging.ipynb``.

非 ``resample`` 语义；窗定义、月界裁剪、覆盖规则与示例见上述教程笔记本。
"""

import numpy as np
import pandas as pd

_MIN_VALID_FRACTION = 0.5  # Strict majority (> half) for coverage checks. / 覆盖判定的严格多数


def _period_delta(rule):
    """
    Map a fixed pandas frequency string to window length Δ.
    将固定 pandas 频率字符串映射为窗长 Δ。

    Parameters
    ----------
    rule : str
        Fixed offset alias (e.g. ``'30min'``, ``'1h'``). Must resolve to a fixed step.
        固定偏移别名（如 ``'30min'``、``'1h'``），须为固定步长。

    Returns
    -------
    pandas.Timedelta
        Window length Δ. / 窗长 Δ。

    Raises
    ------
    ValueError
        If ``rule`` is not a fixed frequency. / ``rule`` 非固定频率时抛出。
    """
    off = pd.tseries.frequencies.to_offset(rule)
    try:
        return pd.Timedelta(off)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"rule {rule!r} must be a fixed frequency (e.g. '30min', '1h'). / "
            "需要固定频率（如 '30min', '1h'）。"
        ) from e


def _archive_timestep_1_or_3(index):
    """
    Infer native timestep as 1 or 3 minutes from median index spacing (BSRN archives).
    由索引步长中位数推断 1 或 3 分钟步长（BSRN 存档惯例）。

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Input timestamps. / 输入时间戳。

    Returns
    -------
    pandas.Timedelta
        ``Timedelta(minutes=1)`` or ``Timedelta(minutes=3)`` (fallback: 1 min).
        ``Timedelta(minutes=1)`` 或 ``Timedelta(minutes=3)``（默认回退 1 分钟）。
    """
    if len(index) < 2:
        return pd.Timedelta(minutes=1)
    d = pd.Series(index).diff().median()
    if pd.isna(d) or d <= pd.Timedelta(0):
        return pd.Timedelta(minutes=1)
    sec = float(d.total_seconds())
    if 0.5 * 60 <= sec <= 1.5 * 60:
        return pd.Timedelta(minutes=1)
    if 2.5 * 60 <= sec <= 3.5 * 60:
        return pd.Timedelta(minutes=3)
    return pd.Timedelta(minutes=1)


def _label_grid(index, rule):
    """
    Build a regular label grid from ``floor(min)`` through ``ceil(max)`` at ``rule``.
    自 ``floor(min)`` 至 ``ceil(max)`` 按 ``rule`` 生成规则标签网格。

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Data extent. / 数据时间范围。
    rule : str
        Bin frequency. / 分箱频率。

    Returns
    -------
    pandas.DatetimeIndex
        Labels inclusive of endpoints; empty if invalid range. / 含端点的标签；范围无效则为空。
    """
    lo = index.min().floor(rule)
    hi = index.max().ceil(rule)
    if lo > hi:
        return pd.DatetimeIndex([], tz=index.tz)
    return pd.date_range(lo, hi, freq=rule, inclusive="both", tz=index.tz)


def _trim_labels_for_alignment(labels, index, rule, alignment, match_ceiling_labels=True):
    """
    Drop edge labels that would mix months (align with floor / ceiling / center grids).
    去掉跨月边界的标签，使与 floor / ceiling / center 网格一致。

    Parameters
    ----------
    labels : pandas.DatetimeIndex
        Full grid from :func:`_label_grid`. / :func:`_label_grid` 的完整网格。
    index : pandas.DatetimeIndex
        Actual data index (defines month span ``lo``, ``hi``). / 实际数据索引（定义月界 ``lo``、``hi``）。
    rule : str
        Bin frequency. / 分箱频率。
    alignment : {'floor', 'ceiling', 'center'}
        Window alignment. / 窗对齐方式。
    match_ceiling_labels : bool, default True
        For ``center`` only: if True, apply ceiling-style trim (``labels > lo``); else floor-style
        (``labels < hi``). / 仅 ``center``：True 时与 ceiling 同裁剪，False 时与 floor 同裁剪。

    Returns
    -------
    pandas.DatetimeIndex
        Trimmed labels. / 裁剪后的标签。

    Raises
    ------
    ValueError
        If ``alignment`` is not recognized. / ``alignment`` 未识别时抛出。
    """
    if len(labels) == 0:
        return labels
    lo = index.min().floor(rule)
    hi = index.max().ceil(rule)
    if alignment == "floor":
        return labels[labels < hi]
    if alignment == "ceiling":
        return labels[labels > lo]
    if alignment == "center":
        if match_ceiling_labels:
            return labels[labels > lo]
        return labels[labels < hi]
    raise ValueError(f"Unknown alignment: {alignment!r}")


def _window_mask(index, L, delta, alignment, resolution):
    """
    Boolean mask: timestamps belonging to the averaging window for label ``L``.
    布尔掩码：属于标签 ``L`` 对应平均窗的时间戳。

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Row timestamps. / 行时间戳。
    L : pandas.Timestamp
        Label (bin anchor). / 标签（分箱锚点）。
    delta : pandas.Timedelta
        Bin length Δ. / 分箱长度 Δ。
    alignment : {'floor', 'ceiling', 'center'}
        Window alignment (see module docstring). / 窗对齐（见模块说明）。
    resolution : pandas.Timedelta
        Native timestep; shifts the **center** window start by ``+resolution``.
        原生步长；**center** 窗左端为 ``L - Δ/2 + resolution``。

    Returns
    -------
    numpy.ndarray
        Boolean array, same length as ``index``. / 与 ``index`` 等长的布尔数组。

    Raises
    ------
    ValueError
        If ``alignment`` is not recognized. / ``alignment`` 未识别时抛出。
    """
    half = delta / 2
    if alignment == "floor":
        return (index >= L) & (index < L + delta)
    if alignment == "ceiling":
        return (index > L - delta) & (index <= L)
    if alignment == "center":
        return (index >= L - half + resolution) & (index <= L + half)
    raise ValueError(f"Unknown alignment: {alignment!r}")


def _expected_timesteps(delta, resolution):
    """
    Nominal number of native steps in one bin (for coverage checks, floor/ceiling).
    单个分箱内名义原生步数（用于 floor/ceiling 覆盖检查）。

    Parameters
    ----------
    delta : pandas.Timedelta
        Bin length. / 分箱长度。
    resolution : pandas.Timedelta
        Native timestep. / 原生步长。

    Returns
    -------
    int
        At least 1. / 至少为 1。
    """
    return max(1, int(round(float(delta / resolution))))


def _count_valid_timesteps(part, df_columns_numeric):
    """
    Count rows with at least one finite numeric value in selected columns.
    统计所选列中至少有一个有限数值的行数。

    Parameters
    ----------
    part : pandas.DataFrame
        Window subset. / 窗口子集。
    df_columns_numeric : list of str
        Column names to test (numeric dtypes). / 要检测的数值列名。

    Returns
    -------
    int
        Row count. / 行数。
    """
    if part.empty:
        return 0
    if len(df_columns_numeric) == 0:
        return len(part)
    return int(part[df_columns_numeric].notna().any(axis=1).sum())


def _nan_row_like(df):
    """
    One row of NaNs with the same columns as ``df`` (placeholder for insufficient coverage).
    与 ``df`` 同列的一行 NaN（覆盖不足时的占位）。

    Parameters
    ----------
    df : pandas.DataFrame
        Template frame. / 模板表。

    Returns
    -------
    pandas.Series
        All-NaN row. / 全 NaN 行。
    """
    return pd.Series(np.nan, index=df.columns)


def _finalize_row(agg, df):
    """
    Normalize aggregation output to a ``Series`` aligned to ``df.columns``.
    将聚合结果规范为与 ``df.columns`` 对齐的 ``Series``。

    Parameters
    ----------
    agg : pandas.Series or scalar or mapping
        Result of ``aggfunc``. / ``aggfunc`` 的返回值。
    df : pandas.DataFrame
        Column order reference. / 列顺序参考。

    Returns
    -------
    pandas.Series
        Reindexed series. / 重索引后的序列。
    """
    if isinstance(agg, pd.Series):
        s = agg
    else:
        s = pd.Series(agg)
    return s.reindex(df.columns)


def _aggregate(part, aggfunc):
    """
    Apply ``aggfunc`` to ``part`` (string name or callable).
    对 ``part`` 应用 ``aggfunc``（字符串名或可调用对象）。

    Parameters
    ----------
    part : pandas.DataFrame
        Window subset. / 窗口子集。
    aggfunc : str or callable
        ``'mean'``, ``'sum'``, ``'median'``, any name accepted by ``DataFrame.agg``, or
        ``callable(DataFrame) -> Series|scalar``.
        ``'mean'``、``'sum'``、``'median'``、``DataFrame.agg`` 支持的名称，或 ``callable``。

    Returns
    -------
    scalar or pandas.Series
        Aggregation result. / 聚合结果。
    """
    if isinstance(aggfunc, str):
        if aggfunc == "mean":
            return part.mean(numeric_only=True)
        if aggfunc == "sum":
            return part.sum(numeric_only=True)
        if aggfunc == "median":
            return part.median(numeric_only=True)
        return part.agg(aggfunc)
    return aggfunc(part)


def pretty_average(df, rule, alignment="ceiling", aggfunc="mean", resolution=None, match_ceiling_labels=True):
    """
    Average ``df`` over explicit labeled windows (not pandas ``resample`` semantics).
    按显式标签窗聚合（非 ``resample``）。语义与示例见 ``docs/tutorials/3.time_averaging.ipynb``。

    Parameters
    ----------
    df : pandas.DataFrame
        Must use a :class:`~pandas.DatetimeIndex`.
        须为 :class:`~pandas.DatetimeIndex`。
    rule : str
        Fixed bin frequency (e.g. ``'1h'``, ``'30min'``). / 固定分箱频率。
    alignment : {'floor', 'ceiling', 'center'}, default ``'ceiling'``
        **floor** ``[L, L+Δ)`` · **ceiling** ``(L-Δ, L]`` · **center** ``[L-Δ/2+res, L+Δ/2]``.
        定义见教程 ``docs/tutorials/3.time_averaging.ipynb``。
    aggfunc : str or callable, default ``'mean'``
        Passed to :func:`_aggregate`. / 传入 :func:`_aggregate`。
    resolution : pandas.Timedelta or None, default None
        Native timestep for **center** windows; if None, inferred as 1 or 3 min via
        :func:`_archive_timestep_1_or_3`. / **center** 窗的原生步长；None 时由 :func:`_archive_timestep_1_or_3` 推断为 1 或 3 分钟。
    match_ceiling_labels : bool, default True
        When ``alignment='center'``, trim monthly edge labels like **ceiling** (default) or **floor**.
        ``alignment='center'`` 时，月界裁剪与 **ceiling**（默认）或 **floor** 一致。

    Returns
    -------
    pandas.DataFrame
        One row per output label; insufficient coverage yields NaN rows (labels retained).
        每个输出标签一行；覆盖不足时保留标签但行为 NaN。

    Raises
    ------
    TypeError
        If ``df.index`` is not a :class:`~pandas.DatetimeIndex`. / 索引非 :class:`~pandas.DatetimeIndex`。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "pretty_average requires a DatetimeIndex; set df.index or use set_index. / "
            "pretty_average 需要 DatetimeIndex。"
        )

    if df.empty:
        return df.copy()

    delta = _period_delta(rule)
    res = resolution if resolution is not None else _archive_timestep_1_or_3(df.index)
    labels = _label_grid(df.index, rule)
    labels = _trim_labels_for_alignment(
        labels, df.index, rule, alignment, match_ceiling_labels=match_ceiling_labels
    )
    if len(labels) == 0:
        return df.iloc[0:0].copy()

    rows = []
    out_labels = []
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    n_exp = _expected_timesteps(delta, res)

    for L in labels:
        mask = _window_mask(df.index, L, delta, alignment, res)
        part = df.loc[mask]
        n_valid = _count_valid_timesteps(part, num_cols)
        if alignment == "center":
            n_den = len(part)
            insufficient = n_den == 0 or n_valid <= n_den * _MIN_VALID_FRACTION
        else:
            insufficient = n_valid <= n_exp * _MIN_VALID_FRACTION

        if insufficient:
            rows.append(_nan_row_like(df))
            out_labels.append(L)
            continue

        rows.append(_finalize_row(_aggregate(part, aggfunc), df))
        out_labels.append(L)

    if not rows:
        return df.iloc[0:0].copy()

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(out_labels, tz=df.index.tz))
    return out
