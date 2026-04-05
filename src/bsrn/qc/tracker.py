"""
BSRN Level 6 checks - tracker-off detection.
BSRN 6 级检查 - 跟踪器失准检测。
"""

import numpy as np
import pandas as pd


def tracker_off_test(ghi, bni, zenith, ghi_extra=None, ghi_clear=None,
                     dhi_clear=None, bni_clear=None):
    """
    Check if the solar tracker is off by comparing measured and clear-sky irradiances [1]_.
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

    Raises
    ------
    ValueError
        If ``ghi_clear`` is not provided and ``ghi_extra`` is also missing.
        未提供 ``ghi_clear`` 且 ``ghi_extra`` 也为空时。

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
