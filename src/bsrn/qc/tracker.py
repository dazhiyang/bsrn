"""
bsrn quality control - tracker-off detection.
BSRN 质量控制 - 跟踪器失准检测。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry, clearsky
from bsrn.constants import BSRN_STATIONS


def tracker_off_test(ghi, bni, zenith, ghi_extra=None, ghi_clear=None, dhi_clear=None, bni_clear=None):
    """
    Check if the solar tracker is off by comparing measured and clear-sky irradiances.
    通过比较测量值和晴空值来检查太阳跟踪器是否失准。

    Parameters
    ----------
    ghi : numeric or Series
        Measured global horizontal irradiance ($G_h$). [W/m^2]
        测量的水平总辐照度 ($G_h$)。[瓦/平方米]
    bni : numeric or Series
        Measured beam normal irradiance ($B_n$). [W/m^2]
        测量的法向直接辐照度 ($B_n$)。[瓦/平方米]
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]
    ghi_extra : numeric or Series, optional
        Extraterrestrial horizontal irradiance ($E_0$). [W/m^2]
        地外水平辐照度 ($E_0$)。[瓦/平方米]
    ghi_clear : numeric or Series, optional
        Reference clear-sky global horizontal irradiance ($G_{hc}$). [W/m^2]
        参考晴空水平总辐照度 ($G_{hc}$)。[瓦/平方米]
    dhi_clear : numeric or Series, optional
        Reference clear-sky diffuse horizontal irradiance ($D_{hc}$). [W/m^2]
        参考晴空水平散射辐照度 ($D_{hc}$)。[瓦/平方米]
    bni_clear : numeric or Series, optional
        Reference clear-sky beam normal irradiance ($B_{nc}$). [W/m^2]
        参考晴空法向直接辐照度 ($B_{nc}$)。[瓦/平方米]

    Returns
    -------
    flags : Series or ndarray
        Boolean flags (True = Pass). [bool]
        布尔标记（True = 通过）。[布尔值]

    References
    ----------
    .. [1] Forstinger, A., et al. (2021). Expert quality control of solar 
       radiation ground data sets. In SWC 2021: ISES Solar World Congress. 
       International Solar Energy Society.
    """
    mu0 = np.cos(np.radians(zenith))
    
    # 1. Fallback definitions per Forstinger et al. (2021) / 按照 Forstinger (2021) 默认定义
    if ghi_clear is None:
        if ghi_extra is None:
            raise ValueError("GHIE (ghi_extra) must be provided if GHIC (ghi_clear) is not supplied. / 如果未提供 GHIC (ghi_clear)，则必须提供 GHIE (ghi_extra)。")
        # GHIC ($G_{hc}$) = 0.8 * GHIE ($E_{0}$) / 晴空水平总辐照度参考值
        ghi_clear = 0.8 * ghi_extra
        
    if dhi_clear is None:
        # DHIC ($D_{hc}$) = 0.165 * GHIC ($G_{hc}$) / 晴空水平散射辐照度参考值
        dhi_clear = 0.165 * ghi_clear
        
    if bni_clear is None:
        # BNIC ($B_{nc}$) = (GHIC - DHIC) / mu0 / 晴空法向直接辐照度参考值
        bni_clear = (ghi_clear - dhi_clear) / np.maximum(mu0, 0.01)
    
    # Tracker-off condition: sunny day where GHI measurement is close to clear-sky but BNI is low
    # 跟踪器失准条件: 晴天 GHI 接近晴空值但 BNI 远低于其参考值
    
    # Term 1: (GHIC - GHI) / (GHIC + GHI) < 0.2 / 条件 1: GHI 接近晴空值
    term1 = (ghi_clear - ghi) / (ghi_clear + ghi)
    
    # Term 2: (BNIC - BNI) / (BNIC + BNI) > 0.95 / 条件 2: BNI 远低于参考值
    term2 = (bni_clear - bni) / (bni_clear + bni)
    
    # Tracker is off if SZA < 85 / 如果天顶角 < 85 则判断跟踪器失准
    tracker_is_off = (term1 < 0.2) & (term2 > 0.95) & (zenith < 85)
    
    if hasattr(tracker_is_off, 'iloc'):
        return ~tracker_is_off
    else:
        return not tracker_is_off


def test_tracker_off(df, station_code=None, lat=None, lon=None, elev=None):
    """
    Detect solar tracker failures on a DataFrame.
    在 DataFrame 上检测太阳跟踪器失准。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data with 'ghi', 'bni'.
        包含 'ghi'、'bni' 的输入 BSRN 数据。
    station_code : str, optional
        BSRN station code to retrieve coordinates.
        用于检索坐标的 BSRN 站点代码。
    lat : float, optional
        Latitude. [degrees] / 纬度。[度]
    lon : float, optional
        Longitude. [degrees] / 经度。[度]
    elev : float, optional
        Elevation. [m] / 海拔。[米]

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'f_tracker_off' flag column.
        增加了 'f_tracker_off' 标记列的 DataFrame。
    """
    if lat is None or lon is None:
        if station_code in BSRN_STATIONS:
            meta = BSRN_STATIONS[station_code]
            lat, lon, elev = meta['lat'], meta['lon'], meta['elev']
        else:
            raise ValueError("Station metadata (lat/lon/elev) must be provided.")

    # Calculate required solar geometry / 计算所需的太阳几何参数
    solpos = geometry.get_solar_position(df.index, lat, lon, elev)
    zenith = solpos["zenith"]
    ghi_extra = geometry.get_ghi_extra(df.index, zenith)

    # Use clear-sky benchmarks if available, else use fallback logic / 如果可用则使用晴空基准，否则使用回退逻辑
    # Note: add_clearsky_columns adds 'ghi_clear', 'bni_clear'
    ghi_c = df['ghi_clear'] if 'ghi_clear' in df.columns else None
    bni_c = df['bni_clear'] if 'bni_clear' in df.columns else None
    
    # Run test / 执行测试
    f_pass = tracker_off_test(df['ghi'], df['bni'], zenith, ghi_extra=ghi_extra, 
                              ghi_clear=ghi_c, bni_clear=bni_c)
    
    df['f_tracker_off'] = (~f_pass).astype(int)

    return df
