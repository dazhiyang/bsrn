"""
bsrn level 3 (inter-comparison) checks - closure test.
BSRN 3 级（相互比较）检查 - 闭合测试。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.constants import BSRN_STATIONS


def closure_low_sza_test(ghi, bni, dhi, zenith):
    r"""
    Check consistency between GHI, BNI, and DHI for low solar zenith angles ($Z \le 75^\circ$).
    检查低太阳天顶角 ($Z \le 75^\circ$) 下 GHI、BNI 和 DHI 之间的一致性。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度 ($G_h$)。[瓦/平方米]
    bni : numeric or Series
        Beam normal irradiance ($B_n$). [W/m^2]
        法向直接辐照度 ($B_n$)。[瓦/平方米]
    dhi : numeric or Series
        Diffuse horizontal irradiance ($D_h$). [W/m^2]
        水平散射辐照度 ($D_h$)。[瓦/平方米]
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]

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
    
    # Calculate GHI from BNI and DHI / 根据 BNI 和 DHI 计算 GHI
    ghi_calc = bni * mu0 + dhi
    ghi_calc_safe = np.where(ghi_calc > 0, ghi_calc, np.nan)
    
    # Condition: |GHI / (DNI * cos(SZA) + DIF) - 1| <= 0.08
    # 条件: |GHI / (DNI * cos(SZA) + DIF) - 1| <= 0.08
    diff_ratio = np.abs(ghi / ghi_calc_safe - 1)
    
    # Domain: Z <= 75 and GHI > 50 / 适用范围: Z <= 75 且 GHI > 50
    in_domain = (zenith <= 75) & (ghi > 50)
    condition_met = diff_ratio <= 0.08
    
    if hasattr(in_domain, 'iloc'):
        return (~in_domain) | condition_met
    else:
        return (not in_domain) or condition_met


def closure_high_sza_test(ghi, bni, dhi, zenith):
    r"""
    Check consistency between GHI, BNI, and DHI for high solar zenith angles ($Z > 75^\circ$).
    检查高太阳天顶角 ($Z > 75^\circ$) 下 GHI、BNI 和 DHI 之间的一致性。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度 ($G_h$)。[瓦/平方米]
    bni : numeric or Series
        Beam normal irradiance ($B_n$). [W/m^2]
        法向直接辐照度 ($B_n$)。[瓦/平方米]
    dhi : numeric or Series
        Diffuse horizontal irradiance ($D_h$). [W/m^2]
        水平散射辐照度 ($D_h$)。[瓦/平方米]
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]

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
    
    # Calculate GHI from BNI and DHI / 根据 BNI 和 DHI 计算 GHI
    ghi_calc = bni * mu0 + dhi
    ghi_calc_safe = np.where(ghi_calc > 0, ghi_calc, np.nan)
    
    # Condition: |GHI / (DNI * cos(SZA) + DIF) - 1| <= 0.15
    # 条件: |GHI / (DNI * cos(SZA) + DIF) - 1| <= 0.15
    diff_ratio = np.abs(ghi / ghi_calc_safe - 1)
    
    # Domain: Z > 75 and GHI > 50 / 适用范围: Z > 75 且 GHI > 50
    in_domain = (zenith > 75) & (ghi > 50)
    condition_met = diff_ratio <= 0.15
    
    if hasattr(in_domain, 'iloc'):
        return (~in_domain) | condition_met
    else:
        return (not in_domain) or condition_met


def test_closure(df, station_code=None, lat=None, lon=None, elev=None):
    """
    Run all Phase 3 (Closure) consistency checks on a DataFrame.
    对 DataFrame 运行所有 3 级（闭合）一致性检查。

    Parameters
    ----------
    df : pd.DataFrame
        Input BSRN data with 'ghi', 'bni', 'dhi'.
        包含 'ghi'、'bni'、'dhi' 的输入 BSRN 数据。
    station_code : str, optional
        BSRN station code to retrieve coordinates.
        用于检索坐标形成 BSRN 站点代码。
    lat : float, optional
        Latitude. [degrees] / 纬度。[度]
    lon : float, optional
        Longitude. [degrees] / 经度。[度]
    elev : float, optional
        Elevation. [m] / 海拔。[米]

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'f_closure' flag column.
        增加了 'f_closure' 标记列的 DataFrame。
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

    # Apply tests / 执行测试 (Low SZA and High SZA tests)
    # We combine them into a single flag column / 将它们合并为一个标记列
    f_low = closure_low_sza_test(df['ghi'], df['bni'], df['dhi'], zenith)
    f_high = closure_high_sza_test(df['ghi'], df['bni'], df['dhi'], zenith)
    
    # Combined flag (True = Pass) / 合并标记（True = 通过）
    df['f_closure'] = (f_low & f_high).astype(int)

    return df
