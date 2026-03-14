"""
clear-sky radiation models.
Provides theoretical reference for QC checks and separation modeling.
晴空辐射模型。
为 QC 检查提供理论参考和直散分离建模。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.constants import BSRN_STATIONS, LINKE_TURBIDITY


def get_relative_airmass(zenith, model='kastenyoung1989'):
    """
    Calculate relative (not pressure-adjusted) airmass at sea level.
    计算海平面处的相对（非气压调整）大气质量。

    Parameters
    ----------
    zenith : numeric
        Zenith angle of the sun. [degrees]
        太阳天顶角。[度]

    model : string, default 'kastenyoung1989'
        Available models include:
        * 'kasten1966' - See [1]_
        * 'kastenyoung1989' (default) - See [2]_

    Returns
    -------
    airmass_relative : numeric
        Relative airmass at sea level. Returns NaN values for any
        zenith angle greater than 90 degrees. [unitless]
        海平面处的相对大气质量。对于任何大于 90 度的天顶角返回 NaN 值。

    References
    ----------
    .. [1] Kasten, F. (1965). A New Table and Approximation Formula for the
       Relative Optical Air Mass (Technical Report 136). Hanover, NH:
       CRREL (U.S. Army).
    .. [2] Kasten, F., & Young, A. T. (1989). Revised optical air mass
       tables and approximation formula. Applied Optics, 28(22), 4735-4738.
    """
    # set zenith values greater than 90 to nans / 将大于 90 的天顶角设为 NaN
    zenith = np.where(zenith > 90, np.nan, zenith)

    model = model.lower()

    if model == 'kastenyoung1989':
        am = (1.0 / (np.cos(np.radians(zenith)) +
              0.50572*((6.07995 + (90 - zenith)) ** - 1.6364)))
    elif model == 'kasten1966':
        am = 1.0 / (np.cos(np.radians(zenith)) + 0.15*((93.885 - zenith) ** - 1.253))
    else:
        raise ValueError(f'{model} is not a valid Kasten model for relative airmass. Use "kastenyoung1989" or "kasten1966".')

    if isinstance(zenith, pd.Series):
        am = pd.Series(am, index=zenith.index)

    return am

def get_absolute_airmass(airmass_relative, pressure=101325.0):
    """
    Calculates absolute (pressure-corrected) airmass.
    计算绝对（经气压校正的）大气质量。

    Parameters
    ----------
    airmass_relative : numeric
        Relative optical air mass. [unitless]
        相对光学大气质量。[无单位]
    pressure : numeric, default 101325.0
        Surface pressure. [Pa]
        地表气压。[帕斯卡]

    Returns
    -------
    airmass_absolute : numeric
        Absolute optical air mass. [unitless]
        绝对光学大气质量。[无单位]
    """
    return airmass_relative * (pressure / 101325.0)

def ineichen_model(apparent_zenith, airmass_absolute, lt, elev, bni_extra):
    """
    Implementation of Ineichen clear-sky model matching the formulation from pvlib.
    与 pvlib 匹配的 Ineichen 晴空模型直接实现。

    Parameters
    ----------
    apparent_zenith : numeric
        Apparent (refraction-corrected) solar zenith angle. [degrees]
        表观（经折射校正的）太阳天顶角。[度]
    airmass_absolute : numeric
        Absolute (pressure-corrected) air mass. [unitless]
        绝对（经气压校正的）大气质量。[无单位]
    lt : numeric
        Linke turbidity factor. [unitless]
        Linke 浑浊因子。[无单位]
    elev : float
        Elevation. [m]
        海拔。[米]
    bni_extra : numeric
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]

    Returns
    -------
    ghi_clear : numeric
        Clear-sky global horizontal irradiance ($G_{hc}$). [W/m^2]
        晴空水平总辐照度 ($G_{hc}$)。[瓦/平方米]
    bni_clear : numeric
        Clear-sky beam normal irradiance ($B_{nc}$). [W/m^2]
        晴空法向直接辐照度 ($B_{nc}$)。[瓦/平方米]
    dhi_clear : numeric
        Clear-sky diffuse horizontal irradiance ($D_{hc}$). [W/m^2]
        晴空水平散射辐照度 ($D_{hc}$)。[瓦/平方米]

    References
    ----------
    .. [1] Ineichen, P., & Perez, R. (2002). A new airmass independent 
       formulation for the Linke turbidity coefficient. Solar Energy, 73(3), 151-157.
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


def threlkeld_jordan_model(zenith, day_of_year):
    """
    Threlkeld-Jordan clear-sky GHI model.
    The published Engerer2 reference uses this for the ktc predictor.
    Threlkeld-Jordan 晴空 GHI 模型；Engerer2 文献采用此模型计算 ktc。

    Parameters
    ----------
    zenith : array-like
        Solar zenith angle. [degrees]
        太阳天顶角。[度]
    day_of_year : array-like
        Day of year. [1–366]
        年积日。[1–366]

    Returns
    -------
    ghi_clear : np.ndarray
        Clear-sky GHI ($G_{hc}$). [W/m^2]
        晴空 GHI ($G_{hc}$)。[瓦/平方米]

    References
    ----------
    .. [1] Threlkeld, J. L., & Jordan, R. C. (1958). Direct solar radiation 
       availability on clear days. ASHRAE Trans, 64(1), 45-105.
    """
    mu0 = np.maximum(np.cos(np.radians(zenith)), 0)
    doy = np.asarray(day_of_year, dtype=float)
    
    # Avoid division by zero when sun is below horizon
    # 避免太阳位于地平线下时除以零
    # Use small floor for stability/ 为稳定性使用小的下限值
    mu0_safe = np.fmax(mu0, 1e-10)

    a_tj = 1160 + 75 * np.sin(np.radians(360 * (doy - 275) / 365))
    k_tj = 0.174 + 0.035 * np.sin(np.radians(360 * (doy - 100) / 365))
    c_tj = 0.095 + 0.04 * np.sin(np.radians(360 * (doy - 100) / 365))

    dni_clear = a_tj * np.exp(-k_tj / mu0_safe)
    ghi_clear = dni_clear * mu0 + c_tj * dni_clear
    
    # Mask night / 掩蔽夜间
    ghi_clear = np.where(mu0 > 0, ghi_clear, 0.0)
    return ghi_clear

def calculate_vapor_pressure(temp_c, rh):
    """
    Calculates actual vapor pressure ($e_a$) in hPa using the Magnus-Tetens formula.
    使用 Magnus-Tetens 公式计算实际水汽压 ($e_a$)，单位为 hPa。

    Parameters
    ----------
    temp_c : numeric
        Air temperature. [°C]
        气温。[摄氏度]
    rh : numeric
        Relative humidity. [%]
        相对湿度。[百分比]

    Returns
    -------
    ea : numeric
        Actual vapor pressure. [hPa]
        实际水汽压。[百帕]

    References
    ----------
    .. [1] Murray, F. W. (1966). On the computation of saturation vapor 
       pressure (Technical Report P3423). Santa Monica, CA: RAND Corp.
    """
    # 1. Calculate Saturation Vapor Pressure (es) / 计算饱和水汽压 (es)
    # 6.112 is the saturation pressure at 0°C in hPa
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    
    # 2. Calculate Actual Vapor Pressure (ea) / 计算实际水汽压 (ea)
    ea = es * (rh / 100.0)
    return ea

def brutsaert_model(temp_c, rh):
    """
    Calculates clear-sky downward longwave radiation ($L_{dc}$) using Brutsaert (1975).
    使用 Brutsaert (1975) 模型计算晴空下行长波辐射 ($L_{dc}$)。

    Parameters
    ----------
    temp_c : numeric
        Air temperature. [°C]
        气温。[摄氏度]
    rh : numeric
        Relative humidity. [%]
        相对湿度。[百分比]

    Returns
    -------
    lwd_clear : numeric
        Clear-sky downward longwave radiation ($L_{dc}$). [W/m^2]
        晴空下行长波辐射 ($L_{dc}$)。[瓦/平方米]

    References
    ----------
    .. [1] Brutsaert, W. (1975). On a derivable formula for long-wave 
       radiation from clear skies. Water Resources Research, 11(5), 742-744.
    """
    # Constants / 常数
    sigma = 5.670373e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
    temp_k = temp_c + 273.15  # Convert to Kelvin / 转换为开尔文
    
    # Get vapor pressure (ea) in hPa / 获取水汽压 (ea)，单位为 hPa
    ea = calculate_vapor_pressure(temp_c, rh)
    
    # Brutsaert (2005) updated emissivity formula / Brutsaert (2005) 更新发射率公式
    # epsilon = 1.323 * (ea / Ta)^(1/7) for ea in hPa (millibars)
    emissivity = 1.323 * (ea / temp_k)**(1/7)
    
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
        BSRN station abbreviation. [e.g., 'QIQ']
        BSRN 站点缩写。[例如 'QIQ']
    model : str, default 'ineichen'
        Clear-sky model to use. ['ineichen', 'mcclear', or 'tj']
        使用的晴空模型。[ 'ineichen'、'mcclear' 或 'tj']

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
    zenith = solpos["zenith"]
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
        am_rel = get_relative_airmass(zenith)
        # Use standard atmosphere scale height for pressure at elevation
        # 使用标准大气标高计算海拔处的测压
        pressure = 101325.0 * np.exp(-elev / 8434.5)
        am_abs = get_absolute_airmass(am_rel, pressure)
        
        # Calculate components / 计算各分量
        ghi_clear, bni_clear, dhi_clear = ineichen_model(apparent_zenith, am_abs, lt_series, elev, bni_extra)
    
    elif model.lower() == "mcclear":
        # Placeholder for McClear model / McClear 模型占位符
        print("Warning: McClear model not yet implemented. Falling back to Ineichen.")
        return add_clearsky_columns(df, station_code, model="ineichen")
    
    elif model.lower() in ("threlkeld_jordan", "tj"):
        # Threlkeld-Jordan (GHI only; for engerer2) / Threlkeld-Jordan（仅 GHI；用于 engerer2）
        doy = df.index.dayofyear.values
        ghi_clear = threlkeld_jordan_model(zenith, doy)
        # BNI and DHI not standard in this simple TJ implementation / 在此简单 TJ 实现中不含 BNI 和 DHI
        bni_clear = np.full_like(ghi_clear, np.nan)
        dhi_clear = np.full_like(ghi_clear, np.nan)
    
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
