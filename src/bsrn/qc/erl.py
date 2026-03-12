"""
bsrn level 2 (extremely rare) checks.
BSRN 2 级（极罕见）检查。。
"""

import numpy as np
import pandas as pd


"""
Citations:
[1] Long, Chuck N., and Yan Shi. "An automated quality assessment and control algorithm for surface radiation 
measurements." Open Atmos. Sci. J 2.1 (2008): 23-37.
"""
def ghi_erl_test(ghi, zenith, bni_extra):
    """
    Check global horizontal irradiance (GHI, $G_h$) against extremely rare limits.
    检查水平总辐照度 (GHI, $G_h$) 是否在极罕见范围内。

    Parameters
    ----------
    ghi : numeric or Series
        global horizontal irradiance ($G_h$) in W/m^2.
        水平总辐照度 ($G_h$)，单位 W/m^2。
    zenith : numeric or Series
        solar zenith angle ($Z$) in degrees.
        太阳天顶角 ($Z$)，单位为度。
    bni_extra : numeric or Series
        extraterrestrial beam normal irradiance ($E_{0n}$) in W/m^2.
        地外法向辐照度 ($E_{0n}$)，单位 W/m^2。

    Returns
    -------
    flags : Series or ndarray
        Boolean flags where True indicates the value is within extremely rare limits.
        布尔标记，True 表示该值在极罕见范围内。
    """
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.maximum(mu0, 0)
    
    # Upper limit / 上限: 1.2 * E_0n * mu0^1.2 + 50
    upper_limit = 1.2 * bni_extra * (mu0 ** 1.2) + 50
    # Lower limit / 下限: -2 W/m^2
    lower_limit = -2
    
    return (ghi >= lower_limit) & (ghi <= upper_limit)


"""
Citations:
[1] Long, Chuck N., and Yan Shi. "An automated quality assessment and control algorithm for surface radiation 
measurements." Open Atmos. Sci. J 2.1 (2008): 23-37.
"""
def bni_erl_test(bni, zenith, bni_extra):
    """
    Check beam normal irradiance (BNI, $B_n$) against extremely rare limits.
    检查法向直接辐照度 (BNI, $B_n$) 是否在极罕见范围内。

    Parameters
    ----------
    bni : numeric or Series
        beam normal irradiance ($B_n$) in W/m^2.
        法向直接辐照度 ($B_n$)，单位 W/m^2。
    zenith : numeric or Series
        solar zenith angle ($Z$) in degrees.
        太阳天顶角 ($Z$)，单位为度。
    bni_extra : numeric or Series
        extraterrestrial beam normal irradiance ($E_{0n}$) in W/m^2.
        地外法向辐照度 ($E_{0n}$)，单位 W/m^2。

    Returns
    -------
    flags : Series or ndarray
        Boolean flags where True indicates the value is within extremely rare limits.
        布尔标记，True 表示该值在极罕见范围内。
    """
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.maximum(mu0, 0)

    # Upper limit / 上限: 0.95 * E_0n * mu0^0.2 + 10
    upper_limit = 0.95 * bni_extra * (mu0 ** 0.2) + 10
    # Lower limit / 下限: -2 W/m^2
    lower_limit = -2
    
    return (bni >= lower_limit) & (bni <= upper_limit)


"""
Citations:
[1] Long, Chuck N., and Yan Shi. "An automated quality assessment and control algorithm for surface radiation 
measurements." Open Atmos. Sci. J 2.1 (2008): 23-37.
"""
def dhi_erl_test(dhi, zenith, bni_extra):
    """
    Check diffuse horizontal irradiance (DHI, $D_h$) against extremely rare limits.
    检查水平散射辐照度 (DHI, $D_h$) 是否在极罕见范围内。

    Parameters
    ----------
    dhi : numeric or Series
        diffuse horizontal irradiance ($D_h$) in W/m^2.
        水平散射辐照度 ($D_h$)，单位 W/m^2。
    zenith : numeric or Series
        solar zenith angle ($Z$) in degrees.
        太阳天顶角 ($Z$)，单位为度。
    bni_extra : numeric or Series
        extraterrestrial beam normal irradiance ($E_{0n}$) in W/m^2.
        地外法向辐照度 ($E_{0n}$)，单位 W/m^2。

    Returns
    -------
    flags : Series or ndarray
        Boolean flags where True indicates the value is within extremely rare limits.
        布尔标记，True 表示该值在极罕见范围内。
    """
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.maximum(mu0, 0)
    
    # Upper limit / 上限: 0.75 * E_0n * mu0^1.2 + 30
    upper_limit = 0.75 * bni_extra * (mu0 ** 1.2) + 30
    # Lower limit / 下限: -2 W/m^2
    lower_limit = -2
    
    return (dhi >= lower_limit) & (dhi <= upper_limit)


"""
Citations:
[1] Long, Chuck N., and Yan Shi. "An automated quality assessment and control algorithm for surface radiation 
measurements." Open Atmos. Sci. J 2.1 (2008): 23-37.
"""
def lwd_erl_test(lwd):
    """
    Check downward longwave radiation (LWD, $L_d$) against extremely rare limits.
    检查下行长波辐射 (LWD, $L_d$) 是否在极罕见范围内。

    Parameters
    ----------
    lwd : numeric or Series
        downward longwave radiation ($L_d$) in W/m^2.
        下行长波辐射 ($L_d$)，单位 W/m^2。

    Returns
    -------
    flags : Series or ndarray
        Boolean flags where True indicates the value is within extremely rare limits.
        布尔标记，True 表示该值在极罕见范围内。
    """
    # Range / 范围: [60, 500] W/m^2
    return (lwd >= 60) & (lwd <= 500)
