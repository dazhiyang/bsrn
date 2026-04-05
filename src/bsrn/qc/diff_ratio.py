"""
BSRN Level 4 checks - diffuse ratio tests (k-tests).
BSRN 4 级检查 - 散射分数测试 (k-测试)。
"""

import numpy as np
import pandas as pd


def k_low_sza_test(ghi, dhi, zenith):
    r"""
    Check diffuse fraction ($k$) for low solar zenith angles ($Z < 75^\circ$) [1]_ [2]_.
    检查低太阳天顶角 ($Z < 75^\circ$) 下的散射分数 ($k$)。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度 ($G_h$)。[瓦/平方米]
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
    .. [1] Forstinger, A., et al. (2021). Expert quality control of solar 
       radiation ground data sets. In SWC 2021: ISES Solar World Congress. 
       International Solar Energy Society.
    .. [2] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    """
    ghi_safe = np.where(ghi > 0, ghi, np.nan)
    k = dhi / ghi_safe
    
    # Domain: Z < 75 and GHI > 50 and k > 0 / 适用范围: Z < 75 且 GHI > 50 且 k > 0
    in_domain = (zenith < 75) & (ghi > 50) & (k > 0)
    condition_met = k < 1.05
    
    if hasattr(in_domain, 'iloc'):
        return (~in_domain) | condition_met
    else:
        return (not in_domain) or condition_met


def k_high_sza_test(ghi, dhi, zenith):
    r"""
    Check diffuse fraction ($k$) for high solar zenith angles ($Z \ge 75^\circ$) [1]_ [2]_.
    检查高太阳天顶角 ($Z \ge 75^\circ$) 下的散射分数 ($k$)。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度 ($G_h$)。[瓦/平方米]
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
    .. [1] Forstinger, A., et al. (2021). Expert quality control of solar 
       radiation ground data sets. In SWC 2021: ISES Solar World Congress. 
       International Solar Energy Society.
    .. [2] Long, C. N., & Shi, Y. (2008). An automated quality assessment 
       and control algorithm for surface radiation measurements. The Open 
       Atmospheric Science Journal, 2(1), 23-37.
    """
    ghi_safe = np.where(ghi > 0, ghi, np.nan)
    k = dhi / ghi_safe
    
    # Domain: Z >= 75 and GHI > 50 and k > 0 / 适用范围: Z >= 75 且 GHI > 50 且 k > 0
    in_domain = (zenith >= 75) & (ghi > 50) & (k > 0)
    condition_met = k < 1.1
    
    if hasattr(in_domain, 'iloc'):
        return (~in_domain) | condition_met
    else:
        return (not in_domain) or condition_met


def k_kt_combined_test(ghi, dhi, bni_extra, zenith):
    """
    Combined check of diffuse fraction ($k$) and clearness index ($k_t$) [1]_.
    结合散射分数 ($k$) 和晴朗指数 ($k_t$) 的测试。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度 ($G_h$)。[瓦/平方米]
    dhi : numeric or Series
        Diffuse horizontal irradiance ($D_h$). [W/m^2]
        水平散射辐照度 ($D_h$)。[瓦/平方米]
    bni_extra : numeric or Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]
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
    .. [1] Forstinger, A., et al. (2021). Expert quality control of solar 
       radiation ground data sets. In SWC 2021: ISES Solar World Congress. 
       International Solar Energy Society.
    """
    mu0 = np.cos(np.radians(zenith))
    kt = ghi / (bni_extra * mu0)
    
    ghi_safe = np.where(ghi > 0, ghi, np.nan)
    k = dhi / ghi_safe
    
    # Domain: kt > 0.6 and GHI > 150 and Z < 85 and k > 0 / 适用范围: kt > 0.6 且 GHI > 150 且 Z < 85 且 k > 0
    in_domain = (kt > 0.6) & (ghi > 150) & (zenith < 85) & (k > 0)
    condition_met = k < 0.96
    
    if hasattr(in_domain, 'iloc'):
        return (~in_domain) | condition_met
    else:
        return (not in_domain) or condition_met
