"""
Quality control summary for BSRN data.
BSRN 数据质量控制摘要。
"""

import numpy as np
import pandas as pd
import bsrn.physics.geometry as geometry
import bsrn.qc.ppl as ppl
import bsrn.qc.erl as erl
import bsrn.qc.closure as closure
import bsrn.qc.k_index as k_index
import bsrn.qc.diff_ratio as diff_ratio
import bsrn.qc.tracker as tracker
from bsrn.modeling.clear_sky import add_clearsky_columns

_BSRN_MISSING_NUMERIC = (-999.0, -99.9, -99.99)
_BSRN_MISSING_ATOL = 5e-3


def _series_not_missing(s):
    """
    True where ``s`` is a measured value (not NaN / not BSRN missing codes).
    非缺失测量。

    Parameters
    ----------
    s : pandas.Series
        Input column.
        输入列。

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``s.index``.
        与 ``s.index`` 对齐的布尔掩码。
    """
    x = pd.to_numeric(s, errors="coerce")
    mask = x.notna().to_numpy(dtype=bool, copy=True)
    vals = x.to_numpy(dtype=np.float64, copy=True)
    for m in _BSRN_MISSING_NUMERIC:
        mask &= ~np.isclose(vals, m, rtol=0.0, atol=_BSRN_MISSING_ATOL, equal_nan=False)
    return pd.Series(mask, index=s.index)


def _series_finite_numeric(s):
    """
    True where ``s`` coerces to a finite float.
    有限浮点。

    Parameters
    ----------
    s : pandas.Series
        Input column.
        输入列。

    Returns
    -------
    pandas.Series
        Boolean mask aligned to ``s.index``.
        与 ``s.index`` 对齐的布尔掩码。
    """
    x = pd.to_numeric(s, errors="coerce")
    v = x.to_numpy(dtype=np.float64, copy=False)
    ok = x.notna().to_numpy(dtype=bool, copy=True) & np.isfinite(v)
    return pd.Series(ok, index=s.index)


def get_daily_stats(df, lat, lon, elev, station_code=None):
    """
    Calculate daily QC statistics and sunshine duration.
    计算每日 QC 统计信息和日照时数。

    Parameters
    ----------
    df : pd.DataFrame
        BSRN data archive.
        BSRN 数据存档。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    elev : float
        Elevation. [m]
        海拔。[米]
    station_code : str, optional
        BSRN station abbreviation. If ``ghi_clear`` / ``bni_clear`` are not on ``df``,
        used with :func:`~bsrn.modeling.clear_sky.add_clearsky_columns` so the
        **tracker** row matches :func:`~bsrn.qc.wrapper.run_qc` (Ineichen references).
        站点缩写。若 ``df`` 上无晴空列，则用于生成与 ``run_qc`` 一致的跟踪器参考。

    Returns
    -------
    daily_df : pd.DataFrame
        Daily counts of flags and sunshine duration metrics.
        每日标记计数和日照时数指标。

    Notes
    -----
    Minutes with BSRN missing codes (``-999``, ``-99.9``, ``-99.99``) or NaN in the
    relevant input column do not add to per-test failure sums. ``is_sunshine`` only
    counts minutes with a valid (non-missing) BNI value above the threshold.
    各测试输入列为 BSRN 缺失码（``-999``、``-99.9``、``-99.99``）或 NaN 的分钟不计入该测试的失败累计；
    ``is_sunshine`` 仅在 BNI 有效且超阈值时计数。

    **TRACKER** uses Ineichen ``ghi_clear`` / ``bni_clear`` like ``run_qc`` (not the
    GHIE-only fallback). Pass ``station_code`` so Linke turbidity matches the registry;
    if only ``lat`` / ``lon`` / ``elev`` are available, clear-sky is still computed but
    turbidity defaults may differ from ``run_qc`` with a named station.
    **TRACKER** 与 ``run_qc`` 相同采用 Ineichen 晴空；建议传 ``station_code`` 以匹配注册表 Linke；仅坐标时浑浊度默认可能与命名站点略有差异。
    """
    # Calculate solar geometry / 计算太阳几何参数
    solpos = geometry.get_solar_position(df.index, lat, lon, elev)
    zenith = solpos["zenith"]
    bni_extra = geometry.get_bni_extra(df.index)
    ghi_extra = geometry.get_ghi_extra(df.index, zenith)

    nm = _series_not_missing
    g, b, d, lwd = df["ghi"], df["bni"], df["dhi"], df["lwd"]
    nm_g, nm_b, nm_d, nm_l = nm(g), nm(b), nm(d), nm(lwd)
    bni_x = pd.Series(bni_extra, index=df.index)
    geo_ok = _series_finite_numeric(zenith) & _series_finite_numeric(bni_x)

    # Sunshine duration (h) / 日照时数 (h)
    # ACT: BNI > 120 W/m^2 / 实际：BNI > 120 W/m^2
    # MAX: Zenith < 90 / 最大：太阳天顶角 < 90
    df["is_sunshine"] = ((df["bni"] > 120) & nm_b).astype(int)
    df["is_daylight"] = (zenith < 90).astype(int)

    # QC Tests (True = Pass, False = Fail) / QC 测试（True = 通过，False = 失败）
    # Failure sums exclude missing / NaN inputs for that test.
    # 失败累计排除该测试相关列的缺失与 NaN。
    df["GHI_PPL"] = (
        (~ppl.ghi_ppl_test(g, zenith, bni_extra)) & nm_g & geo_ok
    ).astype(int)
    df["GHI_ERL"] = (
        (~erl.ghi_erl_test(g, zenith, bni_extra)) & nm_g & geo_ok
    ).astype(int)
    df["DHI_PPL"] = (
        (~ppl.dhi_ppl_test(d, zenith, bni_extra)) & nm_d & geo_ok
    ).astype(int)
    df["DHI_ERL"] = (
        (~erl.dhi_erl_test(d, zenith, bni_extra)) & nm_d & geo_ok
    ).astype(int)
    df["BNI_PPL"] = ((~ppl.bni_ppl_test(b, bni_extra)) & nm_b & geo_ok).astype(int)
    df["BNI_ERL"] = (
        (~erl.bni_erl_test(b, zenith, bni_extra)) & nm_b & geo_ok
    ).astype(int)
    df["LWD_PPL"] = (~ppl.lwd_ppl_test(lwd) & nm_l).astype(int)
    df["LWD_ERL"] = (~erl.lwd_erl_test(lwd) & nm_l).astype(int)

    v3 = nm_g & nm_b & nm_d
    low = closure.closure_low_sza_test(g, b, d, zenith)
    high = closure.closure_high_sza_test(g, b, d, zenith)
    df["CMP_CLO"] = (((~low) | (~high)) & v3).astype(int)

    v2 = nm_g & nm_d
    df["CMP_DIF"] = (
        ((~diff_ratio.k_low_sza_test(g, d, zenith))
         | (~diff_ratio.k_high_sza_test(g, d, zenith)))
        & v2
    ).astype(int)

    fail_kbkt = ~k_index.kb_kt_test(g, b, bni_extra, zenith) & nm_g & nm_b & geo_ok
    fail_kkt = (
        ~diff_ratio.k_kt_combined_test(g, d, bni_extra, zenith) & nm_g & nm_d & geo_ok
    )
    fail_kbl = ~k_index.kb_limit_test(b, bni_extra, elev, g) & nm_b & nm_g & geo_ok
    fail_ktl = ~k_index.kt_limit_test(g, bni_extra, zenith) & nm_g & geo_ok
    df["CMP_K"] = (fail_kbkt | fail_kkt | fail_kbl | fail_ktl).astype(int)

    # Tracker: match run_qc by using Ineichen GHIC/BNIC when clearsky not on df / 与 run_qc 一致：用 Ineichen 晴空
    ghi_clear = df["ghi_clear"] if "ghi_clear" in df.columns else None
    bni_clear = df["bni_clear"] if "bni_clear" in df.columns else None
    if ghi_clear is None or bni_clear is None:
        if station_code is not None:
            _sky = add_clearsky_columns(df.copy(), station_code=station_code)
        else:
            _sky = add_clearsky_columns(df.copy(), lat=lat, lon=lon, elev=elev)
        ghi_clear = _sky["ghi_clear"]
        bni_clear = _sky["bni_clear"]

    vt = nm_g & nm_b & geo_ok
    pass_trk = tracker.tracker_off_test(
        g, b, zenith, ghi_extra=ghi_extra, ghi_clear=ghi_clear, bni_clear=bni_clear
    )
    df["TRACKER"] = ((~pass_trk) & vt).astype(int)

    # Aggregate by day / 按天汇总
    daily = df.groupby(df.index.date).agg({
        'is_sunshine': 'sum',
        'is_daylight': 'sum',
        'GHI_PPL': 'sum', 'GHI_ERL': 'sum',
        'DHI_PPL': 'sum', 'DHI_ERL': 'sum',
        'BNI_PPL': 'sum', 'BNI_ERL': 'sum',
        'LWD_PPL': 'sum', 'LWD_ERL': 'sum',
        'CMP_CLO': 'sum', 'CMP_DIF': 'sum', 'CMP_K': 'sum',
        'TRACKER': 'sum'
    })
    
    daily['SD_ACT'] = daily['is_sunshine'] / 60.0
    daily['SD_MAX'] = daily['is_daylight'] / 60.0
    daily['SD_REL'] = (daily['SD_ACT'] / daily['SD_MAX'] * 100.0).fillna(0)
    
    return daily
