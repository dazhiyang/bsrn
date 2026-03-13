"""
clear-sky radiation models.
Provides theoretical reference for QC checks.
晴空辐射模型。
为 QC 检查提供理论参考。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.constants import BSRN_STATIONS, LINKE_TURBIDITY

"""
Citations:
[1] Kasten, Fritz, and Andrew T. Young. "Revised optical air mass tables and approximation formula." 
Applied Optics 28.22 (1989): 4735-4738.
"""
def get_airmass(zenith):
    """
    Calculates relative optical air mass ($m$) using Kasten and Young (1989).
    使用 Kasten 和 Young (1989) 模型计算相对光学大气质量 ($m$)。

    Parameters
    ----------
    zenith : numeric
        Solar zenith angle ($Z$) in degrees.
        太阳天顶角 ($Z$)，单位为度。

    Returns
    -------
    airmass : numeric
        Relative optical air mass.
        相对光学大气质量。
    """
    zenith = np.where(zenith > 90, 90, zenith)
    # Kasten-Young formula / Kasten-Young 公式
    am = 1.0 / (np.cos(np.radians(zenith)) + 0.50572 * (96.07995 - zenith)**-1.6364)
    return np.where(zenith >= 90, 0, am)

def get_absolute_airmass(airmass_relative, pressure=101325.0):
    """
    Calculates absolute (pressure-corrected) airmass.
    计算绝对（经气压校正的）大气质量。

    Parameters
    ----------
    airmass_relative : numeric
        Relative optical air mass.
        相对光学大气质量。
    pressure : numeric, default 101325.0
        Surface pressure in Pascals.
        地表气压，单位为帕斯卡。

    Returns
    -------
    airmass_absolute : numeric
        Absolute optical air mass.
        绝对光学大气质量。
    """
    return airmass_relative * (pressure / 101325.0)

"""
Citations:
[1] Ineichen, Pierre, and Richard Perez. "A new airmass independent formulation for the Linke 
turbidity coefficient." Solar Energy 73.3 (2002): 151-157.
"""
def ineichen_model(apparent_zenith, airmass_absolute, bni_extra, lt, elev):
    """
    Implementation of Ineichen clear-sky model matching the formulation from pvlib.
    与 pvlib 匹配的 Ineichen 晴空模型直接实现。

    Parameters
    ----------
    apparent_zenith : numeric
        Apparent (refraction-corrected) solar zenith angle in degrees.
        表观（经折射校正的）太阳天顶角，单位为度。
    airmass_absolute : numeric
        Absolute (pressure-corrected) air mass.
        绝对（经气压校正的）大气质量。
    bni_extra : numeric
        Extraterrestrial beam normal irradiance ($E_{0n}$) in W/m^2.
        地外法向辐照度 ($E_{0n}$)，单位 W/m^2。
    lt : numeric
        Linke turbidity factor.
        Linke 浑浊因子。
    elev : float
        Elevation in meters.
        海拔（米）。

    Returns
    -------
    ghi_clear : numeric
        Clear-sky global horizontal irradiance ($G_{hc}$) in W/m^2.
        晴空水平总辐照度 ($G_{hc}$)，单位 W/m^2。
    bni_clear : numeric
        Clear-sky beam normal irradiance ($B_{nc}$) in W/m^2.
        晴空法向直接辐照度 ($B_{nc}$)，单位 W/m^2。
    dhi_clear : numeric
        Clear-sky diffuse horizontal irradiance ($D_{hc}$) in W/m^2.
        晴空水平散射辐照度 ($D_{hc}$)，单位 W/m^2。
    """
    mu0 = np.maximum(np.cos(np.radians(apparent_zenith)), 0)
    
    # Altitude-dependent coefficients / 与海拔相关的系数
    fh1 = np.exp(-elev / 8000.0)
    fh2 = np.exp(-elev / 1250.0)
    cg1 = 0.0000509 * elev + 0.868
    cg2 = 0.0000392 * elev + 0.0387
    
    # GHI calculation / GHI 计算
    ghi_clear = np.exp(-cg2 * airmass_absolute * (fh1 + fh2 * (lt - 1)))
    # apply extraterrestrial scaling and protect against airmass NaNs creating negatives
    ghi_clear = cg1 * bni_extra * mu0 * np.fmax(ghi_clear, 0)
    
    # BNI calculation / BNI 计算 (Approximation based on Ineichen)
    b = 0.664 + 0.163 / fh1
    bni_clear = b * np.exp(-0.09 * airmass_absolute * (lt - 1))
    bni_clear = bni_extra * np.fmax(bni_clear, 0)
    
    # "empirical correction"
    with np.errstate(divide='ignore', invalid='ignore'):
        bni_clear_2 = ((1 - (0.1 - 0.2 * np.exp(-lt)) / (0.1 + 0.882 / fh1)) / mu0)
    
    bni_clear_2 = ghi_clear * np.fmin(np.fmax(bni_clear_2, 0), 1e20)
    
    bni_clear = np.minimum(bni_clear, bni_clear_2)
    
    # DHI by subtraction / 通过差值计算 DHI
    dhi_clear = ghi_clear - bni_clear * mu0
    
    return ghi_clear, bni_clear, dhi_clear

"""
Citations:
[1] Murray, Francis W. On the computation of saturation vapor pressure. No. P3423. 1966.
"""
def calculate_vapor_pressure(temp_c, rh):
    """
    Calculates actual vapor pressure ($e_a$) in hPa using the Magnus-Tetens formula.
    使用 Magnus-Tetens 公式计算实际水汽压 ($e_a$)，单位为 hPa。

    Parameters
    ----------
    temp_c : numeric
        Air temperature in Celsius ($T_c$).
        气温，单位为摄氏度 ($T_c$)。
    rh : numeric
        Relative humidity in percent (0-100).
        相对湿度，以百分比表示 (0-100)。

    Returns
    -------
    ea : numeric
        Actual vapor pressure in hPa.
        实际水汽压，单位为 hPa。
    """
    # 1. Calculate Saturation Vapor Pressure (es) / 计算饱和水汽压 (es)
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    
    # 2. Calculate Actual Vapor Pressure (ea) / 计算实际水汽压 (ea)
    ea = es * (rh / 100.0)
    return ea

"""
Citations:
[1] Brutsaert, Wilfried. "On a derivable formula for long‐wave radiation from clear skies." Water resources research 11.5 (1975): 742-744.
"""
def brutsaert_model(temp_c, rh):
    """
    Calculates clear-sky downward longwave radiation ($L_{dc}$) using Brutsaert (1975).
    使用 Brutsaert (1975) 模型计算晴空下行长波辐射 ($L_{dc}$)。

    Parameters
    ----------
    temp_c : numeric
        Air temperature in Celsius ($T_c$).
        气温，单位为摄氏度 ($T_c$)。
    rh : numeric
        Relative humidity in percent (0-100).
        相对湿度，以百分比表示 (0-100)。

    Returns
    -------
    lwd_clear : numeric
        Clear-sky downward longwave radiation in W/m^2.
        晴空下行长波辐射，单位为 W/m^2。
    """
    # Constants / 常数
    sigma = 5.670373e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
    temp_k = temp_c + 273.15  # Convert to Kelvin / 转换为开尔文
    
    # Get vapor pressure (ea) in hPa / 获取水汽压 (ea)，单位为 hPa
    ea = calculate_vapor_pressure(temp_c, rh)
    
    # Brutsaert (1975) Emissivity formula / Brutsaert (1975) 发射率公式
    # epsilon = 0.643 * (ea / Ta)^(1/7)
    emissivity = 0.643 * (ea / temp_k)**(1/7)
    
    # Calculate Final Downward Radiation (L_down) / 计算最终的下行辐射
    lwd_clear = emissivity * sigma * (temp_k**4)
    
    return lwd_clear


def add_clearsky_columns(df, station_code, model="ineichen"):
    """
    Adds clear-sky radiation columns to a DataFrame based on its DatetimeIndex.
    根据 DatetimeIndex 向 DataFrame 添加晴空辐射列。

    Parameters
    ----------
    df : pd.DataFrame
        Input data with pd.DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    station_code : str
        BSRN station abbreviation (e.g., 'QIQ').
        BSRN 站点缩写（例如 'QIQ'）。
    model : str, default 'ineichen'
        Clear-sky model to use ('ineichen' or 'mcclear').
        使用的晴空模型（'ineichen' 或 'mcclear'）。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added _clear columns.
        增加了 _clear 列的 DataFrame。
    """
    if station_code not in BSRN_STATIONS:
        print(f"Error: Station {station_code} not found in BSRN_STATIONS.")
        return df

    meta = BSRN_STATIONS[station_code]
    lat, lon, elev = meta["lat"], meta["lon"], meta["elev"]
    
    # Get solar geometry / 获取太阳几何参数
    solpos = geometry.get_solar_position(df.index, lat, lon, elev)
    apparent_zenith = solpos["apparent_zenith"]
    bni_extra = geometry.get_bni_extra(df.index)
    
    if model.lower() == "ineichen":
        # Handle monthly LT values / 处理月度 LT 值
        # Broadcast LT based on index months / 根据索引月份广播 LT 值
        lt_mapping = LINKE_TURBIDITY.get(station_code, {m: 3.0 for m in ["Jan", "Feb"]})
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # Map months to LT / 将月份映射到 LT
        months = df.index.month - 1
        lt_series = np.array([lt_mapping[month_names[m]] for m in months])
        
        # Airmass calculations / 大气质量计算
        am_rel = get_airmass(apparent_zenith)
        # Use standard atmosphere scale height for pressure at elevation / 使用标准大气标高计算气压
        pressure = 101325.0 * np.exp(-elev / 8434.5)
        am_abs = get_absolute_airmass(am_rel, pressure)
        
        # Calculate components / 计算各分量
        ghi_clear, bni_clear, dhi_clear = ineichen_model(apparent_zenith, am_abs, bni_extra, lt_series, elev)
    
    elif model.lower() == "mcclear":
        # Placeholder for McClear model / McClear 模型占位符
        print("Warning: McClear model not yet implemented. Falling back to Ineichen.")
        return add_clearsky_columns(df, station_code, model="ineichen")
    
    else:
        print(f"Error: Unknown model {model}. Supported: 'ineichen', 'mcclear'.")
        return df
    
    df["ghi_clear"] = ghi_clear
    df["bni_clear"] = bni_clear
    df["dhi_clear"] = dhi_clear
    
    # Calculate clear-sky LWD if temp and rh are available / 如果 temp 和 rh 可用，则计算晴空 LWD
    if "temp" in df.columns and "rh" in df.columns:
        df["lwd_clear"] = brutsaert_model(df["temp"], df["rh"])
    
    return df
