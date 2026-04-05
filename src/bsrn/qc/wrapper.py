"""
High-level QC runners and metadata helpers for BSRN DataFrames.
面向 BSRN DataFrame 的高层次 QC 运行器与元数据辅助函数。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.constants import BSRN_STATIONS
from . import ppl, erl, closure, diff_ratio, k_index, tracker


def _get_metadata(station_code, lat, lon, elev):
    """
    Resolve lat/lon/elev from explicit values and/or BSRN station registry.
    由显式坐标与/或 BSRN 站点注册表解析纬度、经度、海拔。

    Parameters
    ----------
    station_code : str or None
        BSRN station abbreviation, if used for lookup.
        若用于查找，则为 BSRN 站点缩写。
    lat, lon, elev : float or None
        Explicit coordinates; missing pieces may be filled from the registry.
        显式坐标；缺失项可由注册表补全。

    Returns
    -------
    tuple of float
        ``(lat, lon, elev)`` in degrees and meters.
        度与米为单位的 ``(lat, lon, elev)``。

    Raises
    ------
    ValueError
        If the station is unknown or coordinates are insufficient.
        站点未知或坐标不足时。
    """
    # Case 1: User provided explicit coordinates / 情况 1：用户提供了明确的坐标
    if lat is not None and lon is not None and elev is not None:
        return lat, lon, elev

    # Case 2: User provided a BSRN station code / 情况 2：用户提供了 BSRN 站点代码
    if station_code is not None:
        if station_code in BSRN_STATIONS:
            meta = BSRN_STATIONS[station_code]
            # Use provided values if available, otherwise fallback to registry
            # 如果提供了值则使用，否则使用注册表默认值
            lat = lat if lat is not None else meta['lat']
            lon = lon if lon is not None else meta['lon']
            elev = elev if elev is not None else meta['elev']
            return lat, lon, elev
        else:
            raise ValueError(
                f"Station '{station_code}' not found in BSRN registry. "
                "For non-BSRN stations, you must provide 'lat', 'lon', and 'elev' manually. / "
                f"在 BSRN 注册表中未找到站点 '{station_code}'。对于非 BSRN 站点，必须手动提供经纬度和海拔。"
            )

    # Case 3: No station code and missing coordinates / 情况 3：既无站点代码也缺少坐标
    raise ValueError(
        "Insufficient metadata. Provide a valid BSRN 'station_code' or "
        "explicit 'lat', 'lon', and 'elev'. / 元数据不足。请提供有效的 BSRN 站点代码或明确的经纬度和海拔。"
    )


def run_qc(df, station_code=None, lat=None, lon=None, elev=None,
           tests=('ppl', 'erl', 'closure', 'diff_ratio', 'k_index', 'tracker')):
    r"""
    Run a suite of QC tests on a BSRN DataFrame with optimized geometry calculations [1]_ [2]_.
    使用优化的几何计算对 BSRN DataFrame 运行一系列 QC 测试。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data containing irradiance ($G_h, B_n, D_h, L_d$) and/or 
        meteorological ($T, RH, P$) columns.
        包含辐照度 ($G_h, B_n, D_h, L_d$) 或气象 ($T, RH, P$) 观测列的输入 BSRN 数据。
    station_code : str, optional
        BSRN station code to retrieve coordinates.
        用于检索坐标的 BSRN 站点代码。
    lat : float, optional
        Latitude ($\phi$). [degrees] / 纬度 ($\phi$)。[度]
    lon : float, optional
        Longitude ($\lambda$). [degrees] / 经度 ($\lambda$)。[度]
    elev : float, optional
        Elevation ($H$). [m] / 海拔 ($H$)。[米]
    tests : tuple of str, optional
        List of tests to run (e.g., 'ppl', 'erl', 'closure'). Default is all.
        要运行的测试列表（例如 'ppl'、'erl'、'closure'）。默认为全部。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added QC flag columns (0 = Pass, 1 = Fail).
        增加了 QC 标记列的 DataFrame（0 = 通过，1 = 失败）。

    Raises
    ------
    TypeError
        If ``df`` is not a :class:`~pandas.DataFrame`.
        ``df`` 非 DataFrame 时。
    ValueError
        If the index is not a :class:`~pandas.DatetimeIndex` or metadata resolution fails
        (see :func:`_get_metadata`).
        索引非 DatetimeIndex，或元数据解析失败（见 :func:`_get_metadata`）时。

    References
    ----------
    .. [1] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    .. [2] Forstinger, A., et al. (2021). Expert quality control of solar 
       radiation ground data sets. In SWC 2021: ISES Solar World Congress. 
       International Solar Energy Society.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame. / 输入 'df' 必须是 pandas DataFrame。")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex to calculate solar position. / DataFrame 索引必须是 DatetimeIndex 以计算太阳位置。")

    # 0. Validate metadata / 验证元数据
    try:
        lat, lon, elev = _get_metadata(station_code, lat, lon, elev)
    except ValueError as e:
        raise ValueError(f"QC Metadata Error / QC 元数据错误: {str(e)}") from e

    # 1. Pre-calculate solar geometry / 预先计算太阳几何参数
    solpos = geometry.get_solar_position(df.index, lat, lon, elev)
    zenith = solpos["zenith"]
    bni_extra = geometry.get_bni_extra(df.index)

    # 2. Apply requested tests / 执行请求的测试
    if 'ppl' in tests:
        if 'ghi' in df.columns:
            df['flagPPLGHI'] = (~ppl.ghi_ppl_test(df['ghi'], zenith, bni_extra)).astype(int)
        if 'bni' in df.columns:
            df['flagPPLBNI'] = (~ppl.bni_ppl_test(df['bni'], bni_extra)).astype(int)
        if 'dhi' in df.columns:
            df['flagPPLDHI'] = (~ppl.dhi_ppl_test(df['dhi'], zenith, bni_extra)).astype(int)
        if 'lwd' in df.columns:
            df['flagPPLLWD'] = (~ppl.lwd_ppl_test(df['lwd'])).astype(int)

    if 'erl' in tests:
        if 'ghi' in df.columns:
            df['flagERLGHI'] = (~erl.ghi_erl_test(df['ghi'], zenith, bni_extra)).astype(int)
        if 'bni' in df.columns:
            df['flagERLBNI'] = (~erl.bni_erl_test(df['bni'], zenith, bni_extra)).astype(int)
        if 'dhi' in df.columns:
            df['flagERLDHI'] = (~erl.dhi_erl_test(df['dhi'], zenith, bni_extra)).astype(int)
        if 'lwd' in df.columns:
            df['flagERLLWD'] = (~erl.lwd_erl_test(df['lwd'])).astype(int)

    if 'closure' in tests:
        if all(c in df.columns for c in ['ghi', 'bni', 'dhi']):
            df['flag3lowSZA'] = (~closure.closure_low_sza_test(df['ghi'], df['bni'], df['dhi'], zenith)).astype(int)
            df['flag3highSZA'] = (~closure.closure_high_sza_test(df['ghi'], df['bni'], df['dhi'], zenith)).astype(int)

    if 'diff_ratio' in tests:
        if all(c in df.columns for c in ['ghi', 'dhi']):
            df['flagKKt'] = (~diff_ratio.k_kt_combined_test(df['ghi'], df['dhi'], bni_extra, zenith)).astype(int)
            df['flagKlowSZA'] = (~diff_ratio.k_low_sza_test(df['ghi'], df['dhi'], zenith)).astype(int)
            df['flagKhighSZA'] = (~diff_ratio.k_high_sza_test(df['ghi'], df['dhi'], zenith)).astype(int)

    if 'k_index' in tests:
        if all(c in df.columns for c in ['ghi', 'bni']):
            df['flagKbKt'] = (~k_index.kb_kt_test(df['ghi'], df['bni'], bni_extra, zenith)).astype(int)
            df['flagKb'] = (~k_index.kb_limit_test(df['bni'], bni_extra, elev, df['ghi'])).astype(int)
            df['flagKt'] = (~k_index.kt_limit_test(df['ghi'], bni_extra, zenith)).astype(int)

    if 'tracker' in tests:
        if all(c in df.columns for c in ['ghi', 'bni']):
            ghi_extra = geometry.get_ghi_extra(df.index, zenith)
            ghi_c = df['ghi_clear'] if 'ghi_clear' in df.columns else None
            bni_c = df['bni_clear'] if 'bni_clear' in df.columns else None
            f_pass = tracker.tracker_off_test(df['ghi'], df['bni'], zenith, ghi_extra=ghi_extra, 
                                            ghi_clear=ghi_c, bni_clear=bni_c)
            df['flagTracker'] = (~f_pass).astype(int)

    return df


def test_physically_possible(df, station_code=None, lat=None,
                             lon=None, elev=None):
    """
    Run all Phase 1 (Physically Possible) checks on a DataFrame.
    对 DataFrame 运行所有 1 级（物理可能）检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data. / 输入 BSRN 数据。
    station_code : str, optional
        BSRN station code. / BSRN 站点代码。
    lat, lon, elev : float, optional
        Station coordinates and elevation. / 站点坐标和海拔。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'flagPPL*' flag columns.
        增加了 'flagPPL*' 标记列的 DataFrame。

    Raises
    ------
    TypeError, ValueError
        Same as :func:`run_qc` for the ``ppl`` test subset.
        与 :func:`run_qc` 在仅运行 ``ppl`` 时相同。
    """
    return run_qc(df, station_code, lat, lon, elev, tests=('ppl',))


def test_extremely_rare(df, station_code=None, lat=None,
                        lon=None, elev=None):
    """
    Run all Phase 2 (Extremely Rare) checks on a DataFrame.
    对 DataFrame 运行所有 2 级（极罕见）检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data. / 输入 BSRN 数据。
    station_code : str, optional
        BSRN station code. / BSRN 站点代码。
    lat, lon, elev : float, optional
        Station coordinates and elevation. / 站点坐标和海拔。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'flagERL*' flag columns.
        增加了 'flagERL*' 标记列的 DataFrame。

    Raises
    ------
    TypeError, ValueError
        Same as :func:`run_qc` for the ``erl`` test subset.
        与 :func:`run_qc` 在仅运行 ``erl`` 时相同。
    """
    return run_qc(df, station_code, lat, lon, elev, tests=('erl',))


def test_closure(df, station_code=None, lat=None,
                 lon=None, elev=None):
    """
    Run all Phase 3 (Closure) consistency checks on a DataFrame.
    对 DataFrame 运行所有 3 级（闭合）一致性检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data. / 输入 BSRN 数据。
    station_code : str, optional
        BSRN station code. / BSRN 站点代码。
    lat, lon, elev : float, optional
        Station coordinates and elevation. / 站点坐标和海拔。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'flag3lowSZA' and 'flag3highSZA' flag columns.
        增加了 'flag3lowSZA' 和 'flag3highSZA' 标记列的 DataFrame。

    Raises
    ------
    TypeError, ValueError
        Same as :func:`run_qc` for the ``closure`` test subset.
        与 :func:`run_qc` 在仅运行 ``closure`` 时相同。
    """
    return run_qc(df, station_code, lat, lon, elev, tests=('closure',))


def test_diff_ratio(df, station_code=None, lat=None,
                    lon=None, elev=None):
    """
    Run all Phase 3 Diffuse Ratio (k) consistency checks on a DataFrame.
    对 DataFrame 运行所有 3 级散射分数 (k) 一致性检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data. / 输入 BSRN 数据。
    station_code : str, optional
        BSRN station code. / BSRN 站点代码。
    lat, lon, elev : float, optional
        Station coordinates and elevation. / 站点坐标和海拔。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'flagKKt', 'flagKlowSZA', 'flagKhighSZA' flag columns.
        增加了 'flagKKt'、'flagKlowSZA'、'flagKhighSZA' 标记列的 DataFrame。

    Raises
    ------
    TypeError, ValueError
        Same as :func:`run_qc` for the ``diff_ratio`` test subset.
        与 :func:`run_qc` 在仅运行 ``diff_ratio`` 时相同。
    """
    return run_qc(df, station_code, lat, lon, elev, tests=('diff_ratio',))


def test_k_index(df, station_code=None, lat=None,
                 lon=None, elev=None):
    """
    Run all Phase 3 Radiometric Index (k-index) checks on a DataFrame.
    对 DataFrame 运行所有 3 级辐射指数 (k-index) 检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data. / 输入 BSRN 数据。
    station_code : str, optional
        BSRN station code. / BSRN 站点代码。
    lat, lon, elev : float, optional
        Station coordinates and elevation. / 站点坐标和海拔。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'flagKbKt', 'flagKb', 'flagKt' flag columns.
        增加了 'flagKbKt'、'flagKb'、'flagKt' 标记列的 DataFrame。

    Raises
    ------
    TypeError, ValueError
        Same as :func:`run_qc` for the ``k_index`` test subset.
        与 :func:`run_qc` 在仅运行 ``k_index`` 时相同。
    """
    return run_qc(df, station_code, lat, lon, elev, tests=('k_index',))


def test_tracker_off(df, station_code=None, lat=None,
                     lon=None, elev=None):
    """
    Run Tracker-off detection on a DataFrame.
    运行跟踪器失准检测。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data. / 输入 BSRN 数据。
    station_code : str, optional
        BSRN station code. / BSRN 站点代码。
    lat, lon, elev : float, optional
        Station coordinates and elevation. / 站点坐标和海拔。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'flagTracker' flag column.
        增加了 'flagTracker' 标记列的 DataFrame。

    Raises
    ------
    TypeError, ValueError
        Same as :func:`run_qc` for the ``tracker`` test subset.
        与 :func:`run_qc` 在仅运行 ``tracker`` 时相同。
    """
    return run_qc(df, station_code, lat, lon, elev, tests=('tracker',))
