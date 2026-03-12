"""
solar geometry calculations.
Uses pvlib for high-precision solar position.
太阳几何计算。
使用 pvlib 进行高精度太阳位置计算。
"""

import pvlib
import pandas as pd
import numpy as np


def get_solar_position(times, lat, lon, elev=0):
    r"""
    Calculates solar zenith angle ($Z$) and solar azimuth angle ($\phi$).
    计算太阳天顶角 ($Z$) 和太阳方位角 ($\phi$)。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
        计算对应的时间。
    lat : float
        Latitude in decimal degrees.
        纬度（十进制度）。
    lon : float
        Longitude in decimal degrees.
        经度（十进制度）。
    elev : float, default 0
        Elevation in meters.
        海拔（米）。

    Returns
    -------
    solpos : pd.DataFrame
        DataFrame with columns 'zenith', 'apparent_zenith', 'azimuth'.
        包含 'zenith' ($Z$)、'apparent_zenith'、'azimuth' ($\phi$) 等列的 DataFrame。
    """
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon, elev)
    return solpos


def get_bni_extra(times):
    """
    Calculates extraterrestrial beam normal irradiance ($BNI_E$, $E_{0n}$).
    计算地外法向辐照度 ($BNI_E$, $E_{0n}$)。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
        计算对应的时间。

    Returns
    -------
    bni_extra : pd.Series
        Extraterrestrial beam normal irradiance ($E_{0n}$) in W/m^2.
        地外法向辐照度 ($E_{0n}$)，单位 W/m^2。
    """
    return pvlib.irradiance.get_extra_radiation(times)


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
        Solar zenith angle ($Z$) in degrees.
        太阳天顶角 ($Z$)，单位为度。

    Returns
    -------
    ghi_extra : pd.Series
        Extraterrestrial horizontal irradiance ($E_0$) in W/m^2.
        地外水平辐照度 ($E_0$)，单位 W/m^2。
    """
    bni_extra = get_bni_extra(times)
    mu0 = np.cos(np.radians(zenith))
    return bni_extra * np.maximum(mu0, 0)
