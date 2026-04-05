"""
Cloud enhancement event (CEE) detection utilities.
云增强事件 (CEE) 检测工具函数。

Implements algorithms for detecting periods when measured global
irradiance is significantly higher than a clear-sky reference
(cloud enhancement events).
实现若干检测实测总辐照度显著高于晴空参考（云增强事件）的算法。
"""

import numpy as np
import pandas as pd


def _as_1d_array(x, name, n=None):
    """
    Convert input to 1D float array and optionally validate length.
    将输入转换为一维浮点数组，并在需要时校验长度。

    Parameters
    ----------
    x : array-like
        Input to convert.
        待转换的输入。
    name : str
        Name used in error messages.
        错误信息中使用的名称。
    n : int, optional
        Required length; if set, raise when len(x) != n.
        要求长度；若设置则当 len(x) != n 时抛出异常。

    Returns
    -------
    np.ndarray
        1D float array.
        一维浮点数组。

    Raises
    ------
    ValueError
        When n is set and len(x) != n.
        当 n 已设置且 len(x) != n 时抛出异常。
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and len(arr) != n:
        raise ValueError(f"{name} must have length {n}. / {name} 长度必须为 {n}。")
    return arr


def _cee_to_output(index, cee_flag, method):
    """
    Standardize cloud enhancement detection outputs.
    标准化云增强事件检测输出。

    Parameters
    ----------
    index : pandas.Index
        Output index.
        输出索引。
    cee_flag : array-like
        1=enhancement, 0=non-enhancement, NaN=invalid.
        1 表示云增强，0 表示非增强，NaN 表示无效。
    method : str
        Method name for output column.
        方法名称（输出列）。

    Returns
    -------
    pandas.DataFrame
        Columns: is_enhancement, cee_flag, method.
        列：is_enhancement、cee_flag、method。
    """
    cee = np.asarray(cee_flag, dtype=float)
    is_enhancement = pd.array(cee == 1.0, dtype="boolean")
    is_enhancement[np.isnan(cee)] = pd.NA

    return pd.DataFrame(
        {
            "is_enhancement": is_enhancement,
            "cee_flag": cee,
            "method": method,
        },
        index=index,
    )


def _cee_sliding_triplet(ghi, ghi_clear, zenith, times, *,
                         window_minutes, sdev_threshold, kappa_threshold,
                         zenith_max, method):
    """
    Three-sample rolling window with clear-sky index, relative SD, and zenith gate.
    三点滑动窗口：结合晴空指数 κ、相对标准差与太阳天顶角门限。

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance $G_h$. [W/m^2]
        水平总辐照度 $G_h$。[瓦/平方米]
    ghi_clear : array-like
        Clear-sky global horizontal irradiance $G_{hc}$. [W/m^2]
        晴空水平总辐照度 $G_{hc}$。[瓦/平方米]
    zenith : array-like
        Solar zenith angle $Z$. [degrees]
        太阳天顶角 $Z$。[度]
    times : pandas.DatetimeIndex or array-like
        Timestamps aligned to ``ghi`` (1-minute cadence expected).
        与 ``ghi`` 对齐的时间戳（预期为 1 分钟步长）。
    window_minutes : int
        Lag in minutes for the ``lo`` and ``hi`` samples around the centre minute.
        相对中心时刻前后偏移的分钟数（``lo`` / ``hi`` 样本）。
    sdev_threshold : float
        Minimum relative standard deviation of the three-point window.
        三点窗口相对标准差下限。
    kappa_threshold : float
        Minimum clear-sky index $\\kappa = G_h / G_{hc}$.
        晴空指数 $\\kappa = G_h / G_{hc}$ 下限。
    zenith_max : float
        Maximum solar zenith angle for valid detection. [degrees]
        有效检测允许的最大太阳天顶角。[度]
    method : str
        Value for the ``method`` column in the output DataFrame.
        输出 DataFrame 中 ``method`` 列的取值。

    Returns
    -------
    pandas.DataFrame
        Standardized CEE output (``is_enhancement``, ``cee_flag``, ``method``).
        标准化云增强检测输出（``is_enhancement``、``cee_flag``、``method``）。
    """
    # Coerce inputs to aligned 1D float arrays / 将输入对齐为一维浮点数组
    ghi = _as_1d_array(ghi, "ghi")
    ghi_clear = _as_1d_array(ghi_clear, "ghi_clear", n=len(ghi))
    zenith = _as_1d_array(zenith, "zenith", n=len(ghi))

    idx = pd.DatetimeIndex(times)
    if len(idx) != len(ghi):
        # Timestamps must match irradiance length / 时间戳长度须与辐照度一致
        raise ValueError("times length must match inputs. / times 长度必须与输入一致。")

    # Rolling triplet: t-lag, t, t+lag (minutes) / 滑动三点：前、中、后（分钟）
    s = pd.Series(ghi, index=idx)
    s_lo = s.shift(window_minutes)
    s_hi = s.shift(-window_minutes)
    window_df = pd.DataFrame({"lo": s_lo, "t": s, "hi": s_hi})

    # Per-row finite count, mean, sample std (ddof=1) / 逐行：有限个数、均值、样本标准差
    n_valid = np.sum(np.isfinite(window_df.to_numpy(dtype=float)), axis=1)
    mean_val = window_df.mean(axis=1, skipna=False)
    sdev = window_df.std(axis=1, ddof=1, skipna=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        sdev_rel = (sdev / mean_val).to_numpy(dtype=float)

    # Clear-sky index κ = G_h / G_{hc}; suppress div/invalid warnings
    # 晴空指数 κ = G_h/G_hc；抑制除零与无效值警告
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = ghi / ghi_clear
    # Threshold masks for κ, zenith, relative SD / κ、天顶角、相对标准差的阈值掩码
    cond_kappa = kappa > kappa_threshold
    cond_zen = zenith < zenith_max
    cond_sdev = sdev_rel > sdev_threshold

    n = len(ghi)
    cee_flag = np.zeros(n, dtype=float)
    # Core validity: full triplet + finite inputs + positive window mean
    # 核心有效性：三点齐、输入有限、窗口均值为正
    valid_core = (
        (n_valid == 3)
        & np.isfinite(ghi)
        & np.isfinite(ghi_clear)
        & np.isfinite(zenith)
        & (mean_val.to_numpy(dtype=float) > 0.0)
    )
    # All criteria together / 全部判据同时满足
    cond_all = cond_sdev & cond_kappa & cond_zen & valid_core
    cee_flag[cond_all] = 1.0
    # Invalid core rows stay NaN (not evaluated as non-CEE) / 核心无效行保持 NaN（不单记为非 CEE）
    cee_flag[~valid_core] = np.nan

    return _cee_to_output(idx, cee_flag, method)


def killinger_ced(ghi, ghi_clear, zenith, times):
    """
    Detect cloud enhancement events using Killinger et al. (2017) [1]_.
    使用 Killinger 等 (2017) 判据（与 :func:`yang_ced` 不同的固定参数）检测云增强事件。

    Parameters
    ----------
    ghi : array-like
        1‑minute global horizontal irradiance. [W/m^2]
        1 分钟水平总辐照度。[瓦/平方米]
    ghi_clear : array-like
        1‑minute clear-sky global horizontal irradiance. [W/m^2]
        1 分钟晴空水平总辐照度。[瓦/平方米]
    zenith : array-like
        1‑minute solar zenith angle. [degrees]
        1 分钟太阳天顶角。[度]
    times : pandas.DatetimeIndex or array-like
        1‑minute timestamps convertible to ``DatetimeIndex``.
        可转换为 ``DatetimeIndex`` 的 1 分钟时间戳。

    Returns
    -------
    pandas.DataFrame
        Standardized CEE output (``method`` column ``\"killinger\"``).

    References
    ----------
    .. [1] Killinger, S., Engerer, N., & Müller, B. (2017). QCPV: A quality
       control algorithm for distributed photovoltaic array power output.
       Solar Energy, 143, 120–131.
    """
    # Killinger-style defaults: shorter lag, looser zenith cap, lower kappa gate vs. Yang.
    # Killinger 默认：更短滞后、更宽天顶角、更低 kappa 阈值（与 Yang 不同）。
    return _cee_sliding_triplet(
        ghi,
        ghi_clear,
        zenith,
        times,
        window_minutes=5,
        sdev_threshold=0.05,
        kappa_threshold=1.05,
        zenith_max=85.0,
        method="killinger",
    )


def yang_ced(ghi, ghi_clear, zenith, times):
    """
    Detect cloud enhancement events using Yang et al. (2018) [1]_.
    使用 Yang 等 (2018) 方法检测云增强事件。

    Parameters
    ----------
    ghi : array-like
        1‑minute global horizontal irradiance. [W/m^2]
        1 分钟水平总辐照度。[瓦/平方米]
    ghi_clear : array-like
        1‑minute clear-sky global horizontal irradiance. [W/m^2]
        1 分钟晴空水平总辐照度。[瓦/平方米]
    zenith : array-like
        1‑minute solar zenith angle. [degrees]
        1 分钟太阳天顶角。[度]
    times : pandas.DatetimeIndex or array-like
        1‑minute timestamps convertible to ``DatetimeIndex``.
        可转换为 ``DatetimeIndex`` 的 1 分钟时间戳。

    Returns
    -------
    out : pandas.DataFrame
        DataFrame indexed by aggregated timestamps (e.g. 6‑minute) with:

        * ``is_enhancement`` – True where a cloud enhancement is detected.
        * ``ced_flag`` – 1 for enhancement, 0 otherwise, NaN when invalid.

        以聚合时间戳（如 6 分钟）为索引的 DataFrame，包含：

        * ``is_enhancement`` – 检测到云增强时为 True。
        * ``ced_flag`` – 1 表示云增强，0 表示非增强，无效为 NaN。

    References
    ----------
    .. [1] Yang, D., Yagli, G. M., & Quan, H. (2018, May). Quality control for solar
       irradiance data. In *2018 IEEE Innovative Smart Grid Technologies-Asia (ISGT Asia)*
       (pp. 208–213). IEEE.
    """
    return _cee_sliding_triplet(
        ghi,
        ghi_clear,
        zenith,
        times,
        window_minutes=6,
        sdev_threshold=0.05,
        kappa_threshold=1.10,
        zenith_max=75.0,
        method="yang",
    )


def gueymard_ced(ghi, ghi_extra, times=None):
    """
    Detect cloud enhancement events when GHI exceeds extraterrestrial GHI [1]_.
    当 GHI 大于地外水平辐照度时检测云增强事件。

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance. [W/m^2]
        水平总辐照度。[瓦/平方米]
    ghi_extra : array-like
        Horizontal extraterrestrial irradiance. [W/m^2]
        地外水平辐照度。[瓦/平方米]
    times : array-like or pandas.DatetimeIndex, optional
        Time index for outputs. If None, a simple RangeIndex is used.
        输出时间索引；若为 None，则使用 RangeIndex。

    Returns
    -------
    pandas.DataFrame
        Standardized CEE output with columns `is_enhancement`, `cee_flag`, `method`.
        标准化 CEE 输出，包含 `is_enhancement`、`cee_flag`、`method`。

    References
    ----------
    .. [1] Gueymard, C. A. (2017). Cloud and albedo enhancement impacts on
       solar irradiance using high-frequency measurements from thermopile and
       photodiode radiometers. Part 1: Impacts on global horizontal irradiance.
       Solar Energy, 153, 755–765.
    """
    ghi = _as_1d_array(ghi, "ghi")
    ghi_extra = _as_1d_array(ghi_extra, "ghi_extra", n=len(ghi))

    if times is None:
        idx = pd.RangeIndex(start=0, stop=len(ghi), step=1, name="time")
    elif isinstance(times, pd.DatetimeIndex):
        if len(times) != len(ghi):
            raise ValueError("times length must match inputs. / times 长度必须与输入一致。")
        idx = times
    else:
        idx = pd.DatetimeIndex(times)
        if len(idx) != len(ghi):
            raise ValueError("times length must match inputs. / times 长度必须与输入一致。")

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi / ghi_extra
    cee_flag = np.zeros(len(ghi), dtype=float)
    cee_flag[kt > 1.0] = 1.0
    bad = ~np.isfinite(ghi) | ~np.isfinite(ghi_extra)
    cee_flag[bad] = np.nan

    return _cee_to_output(idx, cee_flag, "gueymard")


def detect_cee(method, **kwargs):
    """
    Wrapper for cloud enhancement event (CEE) detection methods.
    云增强事件检测方法封装函数。

    Parameters
    ----------
    method : {"killinger", "yang", "gueymard"}
        Method name.
        方法名称。
    **kwargs : dict
        Arguments forwarded to method function.
        传递给具体方法函数的参数。

    Returns
    -------
    pandas.DataFrame
        Standardized CEE output with is_enhancement, cee_flag, method, etc.
        标准化 CEE 输出，包含 is_enhancement、cee_flag、method 等列。

    Raises
    ------
    ValueError
        When method is not one of the supported CEE methods.
        当方法名不在支持列表中时抛出异常。
    """
    method_key = str(method).strip().lower()
    if method_key == "killinger":
        return killinger_ced(**kwargs)
    if method_key == "yang":
        return yang_ced(**kwargs)
    if method_key == "gueymard":
        return gueymard_ced(**kwargs)
    raise ValueError(
        "Unsupported method. Choose one of killinger, yang, gueymard. / "
        "不支持的方法，请选择 killinger、yang、gueymard。"
    )

