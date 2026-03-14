"""
bsrn level 3 (inter-comparison) checks - closure test.
BSRN 3 级（相互比较）检查 - 闭合测试。
"""

import numpy as np
import pandas as pd


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
