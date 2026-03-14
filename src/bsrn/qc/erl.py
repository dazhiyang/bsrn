"""
bsrn level 2 (extremely rare) checks.
BSRN 2 级（极罕见）检查。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.constants import BSRN_STATIONS


def ghi_erl_test(ghi, zenith, bni_extra):
    """
    Check global horizontal irradiance (GHI, $G_h$) against extremely rare limits.
    检查水平总辐照度 (GHI, $G_h$) 是否在极罕见范围内。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度 ($G_h$)。[瓦/平方米]
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]
    bni_extra : numeric or Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]

    Returns
    -------
    flags : Series or ndarray
        Boolean flags (True = Pass). [bool]
        布尔标记（True = 通过）。[布尔值]

    References
    ----------
    .. [1] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    """
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.maximum(mu0, 0)
    
    # Upper limit: 1.2 * E_0n * mu0^1.2 + 50 / 上限: 1.2 * E_0n * mu0^1.2 + 50
    upper_limit = 1.2 * bni_extra * (mu0 ** 1.2) + 50
    # Lower limit: -2 W/m^2 / 下限: -2 W/m^2
    lower_limit = -2
    
    return (ghi >= lower_limit) & (ghi <= upper_limit)


def bni_erl_test(bni, zenith, bni_extra):
    """
    Check beam normal irradiance (BNI, $B_n$) against extremely rare limits.
    检查法向直接辐照度 (BNI, $B_n$) 是否在极罕见范围内。

    Parameters
    ----------
    bni : numeric or Series
        Beam normal irradiance ($B_n$). [W/m^2]
        法向直接辐照度 ($B_n$)。[瓦/平方米]
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]
    bni_extra : numeric or Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]

    Returns
    -------
    flags : Series or ndarray
        Boolean flags (True = Pass). [bool]
        布尔标记（True = 通过）。[布尔值]

    References
    ----------
    .. [1] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    """
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.maximum(mu0, 0)

    # Upper limit: 0.95 * E_0n * mu0^0.2 + 10 / 上限: 0.95 * E_0n * mu0^0.2 + 10
    upper_limit = 0.95 * bni_extra * (mu0 ** 0.2) + 10
    # Lower limit: -2 W/m^2 / 下限: -2 W/m^2
    lower_limit = -2
    
    return (bni >= lower_limit) & (bni <= upper_limit)


def dhi_erl_test(dhi, zenith, bni_extra):
    """
    Check diffuse horizontal irradiance (DHI, $D_h$) against extremely rare limits.
    检查水平散射辐照度 (DHI, $D_h$) 是否在极罕见范围内。

    Parameters
    ----------
    dhi : numeric or Series
        Diffuse horizontal irradiance ($D_h$). [W/m^2]
        水平散射辐照度 ($D_h$)。[瓦/平方米]
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]
    bni_extra : numeric or Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]

    Returns
    -------
    flags : Series or ndarray
        Boolean flags (True = Pass). [bool]
        布尔标记（True = 通过）。[布尔值]

    References
    ----------
    .. [1] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    """
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.maximum(mu0, 0)
    
    # Upper limit: 0.75 * E_0n * mu0^1.2 + 30 / 上限: 0.75 * E_0n * mu0^1.2 + 30
    upper_limit = 0.75 * bni_extra * (mu0 ** 1.2) + 30
    # Lower limit: -2 W/m^2 / 下限: -2 W/m^2
    lower_limit = -2
    
    return (dhi >= lower_limit) & (dhi <= upper_limit)


def lwd_erl_test(lwd):
    """
    Check downward longwave radiation (LWD, $L_d$) against extremely rare limits.
    检查下行长波辐射 (LWD, $L_d$) 是否在极罕见范围内。

    Parameters
    ----------
    lwd : numeric or Series
        Downward longwave radiation ($L_d$). [W/m^2]
        下行长波辐射 ($L_d$)。[瓦/平方米]

    Returns
    -------
    flags : Series or ndarray
        Boolean flags (True = Pass). [bool]
        布尔标记（True = 通过）。[布尔值]

    References
    ----------
    .. [1] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    """
    # Range: [60, 500] W/m^2 / 范围: [60, 500] W/m^2
    return (lwd >= 60) & (lwd <= 500)


def test_extremely_rare(df, station_code=None, lat=None, lon=None, elev=None):
    """
    Run all Phase 2 (Extremely Rare) checks on a DataFrame.
    对 DataFrame 运行所有 2 级（极罕见）检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data with 'ghi', 'bni', 'dhi', 'lwd'.
        包含 'ghi'、'bni'、'dhi'、'lwd' 的输入 BSRN 数据。
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
        DataFrame with added 'f_erl_ghi', 'f_erl_bni', 'f_erl_dhi', 'f_erl_lwd' flag columns.
        增加了 'f_erl_ghi'、'f_erl_bni'、'f_erl_dhi'、'f_erl_lwd' 标记列的 DataFrame。
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
    bni_extra = geometry.get_bni_extra(df.index)

    # Apply tests / 执行测试
    if 'ghi' in df.columns:
        df['f_erl_ghi'] = ghi_erl_test(df['ghi'], zenith, bni_extra).astype(int)
    if 'bni' in df.columns:
        df['f_erl_bni'] = bni_erl_test(df['bni'], zenith, bni_extra).astype(int)
    if 'dhi' in df.columns:
        df['f_erl_dhi'] = dhi_erl_test(df['dhi'], zenith, bni_extra).astype(int)
    if 'lwd' in df.columns:
        df['f_erl_lwd'] = lwd_erl_test(df['lwd']).astype(int)

    return df
