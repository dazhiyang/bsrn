"""
bsrn quality control - tracker-off detection.
BSRN 质量控制 - 跟踪器失准检测。
"""

import numpy as np
import pandas as pd


"""
Citations:
[1] Forstinger, Anne, et al. "Expert quality control of solar radiation ground data sets." SWC 2021: 
ISES Solar World Congress. International Solar Energy Society, 2021.
"""
def tracker_off_test(ghi, bni, zenith, ghi_extra=None, ghi_clear=None, dhi_clear=None, bni_clear=None):
    """
    Check if the solar tracker is off by comparing measured and clear-sky irradiances.
    通过比较测量值和晴空值来检查太阳跟踪器是否失准。

    Parameters
    ----------
    ghi : numeric or Series
        measured global horizontal irradiance ($G_h$) in W/m^2.
        测量的水平总辐照度 ($G_h$)，单位 W/m^2。
    bni : numeric or Series
        measured beam normal irradiance ($B_n$) in W/m^2.
        测量的法向直接辐照度 ($B_n$)，单位 W/m^2。
    zenith : numeric or Series
        solar zenith angle ($Z$) in degrees.
        太阳天顶角 ($Z$)，单位为度。
    ghi_extra : numeric or Series, optional
        extraterrestrial horizontal irradiance ($E_0$) in W/m^2. Used for default clear calculations.
        地外水平辐照度 ($E_0$)，单位 W/m^2。用于默认的晴空计算。
    ghi_clear : numeric or Series, optional
        reference clear-sky global horizontal irradiance ($G_{hc}$) in W/m^2.
        参考晴空水平总辐照度 ($G_{hc}$)，单位 W/m^2。
    dhi_clear : numeric or Series, optional
        reference clear-sky diffuse horizontal irradiance ($D_{hc}$) in W/m^2.
        参考晴空水平散射辐照度 ($D_{hc}$)，单位 W/m^2。
    bni_clear : numeric or Series, optional
        reference clear-sky beam normal irradiance ($B_{nc}$) in W/m^2.
        参考晴空法向直接辐照度 ($B_{nc}$)，单位 W/m^2。

    Returns
    -------
    flags : Series or ndarray
        Boolean flags where True indicates the tracker is functioning correctly (not off).
        布尔标记，True 表示跟踪器正常运行（未失准）。
    """
    mu0 = np.cos(np.radians(zenith))
    
    # 1. Fallback definitions per Forstinger et al. (2021) / 按照 Forstinger (2021) 默认定义
    if ghi_clear is None:
        if ghi_extra is None:
            raise ValueError("GHIE (ghi_extra) must be provided if GHIC (ghi_clear) is not supplied. / 如果未提供 GHIC (ghi_clear)，则必须提供 GHIE (ghi_extra)。")
        # GHIC ($G_{hc}$) = 0.8 * GHIE ($E_{0}$)
        ghi_clear = 0.8 * ghi_extra
        
    if dhi_clear is None:
        # DHIC ($D_{hc}$) = 0.165 * GHIC ($G_{hc}$)
        dhi_clear = 0.165 * ghi_clear
        
    if bni_clear is None:
        # BNIC ($B_{nc}$) = (GHIC - DHIC) / mu0
        bni_clear = (ghi_clear - dhi_clear) / np.maximum(mu0, 0.01)
    
    # Tracker-off Condition / 跟踪器失准条件:
    # A sunny day where GHI measurement is high (close to clear-sky) but BNI is low / 晴天 GHI 高但 BNI 低
    
    # Conditions per reference image / 根据参考图的条件:
    # Term 1: (GHIC - GHI) / (GHIC + GHI) < 0.2
    term1 = (ghi_clear - ghi) / (ghi_clear + ghi)
    
    # Term 2: (BNIC - BNI) / (BNIC + BNI) > 0.95
    term2 = (bni_clear - bni) / (bni_clear + bni)
    
    # 3. SZA < 85
    tracker_is_off = (term1 < 0.2) & (term2 > 0.95) & (zenith < 85)
    
    if hasattr(tracker_is_off, 'iloc'):
        return ~tracker_is_off
    else:
        return not tracker_is_off
