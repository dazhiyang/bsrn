"""
solar geometry calculations.
Provides high-precision solar position and extraterrestrial radiation.
太阳几何计算。
提供高精度太阳位置和地外辐射计算。
"""

import pandas as pd
import numpy as np
from bsrn.physics import spa


def get_solar_position(times, lat, lon, elev=0):
    r"""
    Calculates solar zenith angle ($Z$) and solar azimuth angle ($\phi$) using SPA.
    使用 SPA 算法计算太阳天顶角 ($Z$) 和太阳方位角 ($\phi$)。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
        计算对应的时间。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    elev : float, default 0
        Elevation. [m]
        海拔。[米]

    Returns
    -------
    solpos : pd.DataFrame
        DataFrame with columns 'zenith' ($Z$), 'apparent_zenith', 'azimuth' ($\phi$). [degrees]
        包含 'zenith' ($Z$)、'apparent_zenith'、'azimuth' ($\phi$) 的 DataFrame。[度]

    References
    ----------
    .. [1] Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: 
       A python package for modeling solar energy systems. Journal of Open 
       Source Software, 3(29), 884.
    .. [2] Anderson, K. S., et al. (2023). pvlib python: 2023 project update. 
       Journal of Open Source Software, 8(92), 5994.
    """
    # Convert times to unix timestamp / 将时间转换为 unix 时间轴
    unixtime = times.view(np.int64) / 1e9
    
    # Use fixed delta_t (69.1s for 2024) for simplicity
    # 为简便起见，使用固定的 delta_t（2024 年约为 69.1 秒）
    zenith, apparent_zenith, azimuth, _ = spa.solar_position(
        unixtime, lat, lon, elev, delta_t=69.1
    )
    
    solpos = pd.DataFrame({
        'zenith': zenith,
        'apparent_zenith': apparent_zenith,
        'azimuth': azimuth
    }, index=times)
    
    return solpos


def get_bni_extra(times):
    """
    Calculates extraterrestrial beam normal irradiance ($BNI_E$, $E_{0n}$) using Spencer (1971).
    使用 Spencer (1971) 方法计算地外法向辐照度 ($BNI_E$, $E_{0n}$)。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
        计算对应的时间。

    Returns
    -------
    bni_extra : pd.Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]

    References
    ----------
    .. [1] Spencer, J. W. (1971). Fourier series representation of the sun. 
       Search, 2, 172.
    """
    # Day angle (radians) using Spencer (1971): (2*pi/365)*(doy - 1)
    # 使用 Spencer (1971) 计算日角（弧度）：(2*pi/365)*(doy - 1)
    day_of_year = times.dayofyear
    b = (2.0 * np.pi / 365.0) * (day_of_year - 1.0)
    
    # Eccentricity correction / 偏心率修正
    # E0 = Isc * (1.00011 + 0.034221*cos(B) + 0.001280*sin(B) + 0.000719*cos(2B) + 0.000077*sin(2B))
    r_r0_sq = (1.00011 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b) + 
               0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))
    
    # Using the project defined solar constant
    # 使用项目定义的太阳常数
    from bsrn.constants import solar_constant
    bni_extra = solar_constant * r_r0_sq
    
    return pd.Series(bni_extra, index=times)


def get_ghi_extra(times, zenith):
    """
    Calculates extraterrestrial horizontal irradiance ($GHI_E$, $E_0$).
    计算地外水平辐照度 ($GHI_E$, $E_0$)。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
        计算对应的时间。
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角 ($Z$)。[度]

    Returns
    -------
    ghi_extra : pd.Series
        Extraterrestrial horizontal irradiance ($E_0$). [W/m^2]
        地外水平辐照度 ($E_0$)。[瓦/平方米]
    """
    bni_extra = get_bni_extra(times)
    mu0 = np.cos(np.radians(zenith))
    return bni_extra * np.maximum(mu0, 0)
