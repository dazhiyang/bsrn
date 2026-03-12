"""
clear-sky radiation models.
Provides theoretical reference for QC checks.
晴空辐射模型。
为 QC 检查提供理论参考。
"""

import pvlib

def get_clearsky_ineichen(times, lat, lon, elev=0, linke_turbidity=3):
    """
    Calculates clear-sky irradiance using the Ineichen-Perez model.
    使用 Ineichen-Perez 模型计算晴空辐照度。

    Parameters
    ----------
    times : pd.DatetimeIndex
        times for calculation.
        计算对应的时间。
    lat : float
        latitude.
        纬度。
    lon : float
        longitude.
        经度。
    elev : float, default 0
        elevation in meters.
        海拔（米）。
    linke_turbidity : float, default 3
        Linke turbidity factor.
        Linke 浑浊因子。

    Returns
    -------
    clearsky : pd.DataFrame
        DataFrame with columns 'ghi_clear', 'bni_clear', 'dhi_clear'.
        包含 'ghi_clear' (晴空水平总辐照度, $G_{hc}$)、'bni_clear' (晴空法向直接辐照度, $B_{nc}$)、'dhi_clear' (晴空水平散射辐照度, $D_{hc}$) 的 DataFrame。
    """
    location = pvlib.location.Location(lat, lon, altitude=elev)
    clearsky = location.get_clearsky(times, model='ineichen', linke_turbidity=linke_turbidity)
    
    # Rename columns to match SKILL.md standards / 重命名列以符合 SKILL.md 标准
    clearsky = clearsky.rename(columns={
        'ghi': 'ghi_clear',
        'dni': 'bni_clear',
        'dhi': 'dhi_clear'
    })
    return clearsky
