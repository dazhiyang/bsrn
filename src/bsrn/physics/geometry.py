"""
solar geometry calculations.
Provides high-precision solar position and extraterrestrial radiation.
太阳几何计算。
提供高精度太阳位置和地外辐射计算。
"""

import pandas as pd
import numpy as np
from bsrn.physics import spa
from bsrn.constants import BSRN_STATIONS


def get_pressure_from_elevation(elev):
    """
    Calculates standard-atmosphere pressure from elevation using pvlib-equivalent formula [1]_.
    使用与 pvlib 等价的公式，根据海拔计算标准大气压。

    Parameters
    ----------
    elev : numeric
        Elevation above sea level. [m]
        海拔高度。[米]

    Returns
    -------
    pressure : numeric
        Surface pressure. [Pa]
        地表气压。[帕]

    References
    ----------
    .. [1] Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python:
       A python package for modeling solar energy systems. Journal of Open Source Software,
       3(29), 884.
    """
    elev_arr = np.asarray(elev, dtype=float)
    if np.any(~np.isfinite(elev_arr)):
        raise ValueError("elev must be finite. / elev 必须为有限值。")

    # Keep real-valued output and fail early for out-of-range altitudes.
    # 保持实数输出，并对超范围海拔提前报错。
    if np.any(elev_arr >= 44331.514):
        raise ValueError("elev must be < 44331.514 m for standard-atmosphere formula.")

    pressure = 100.0 * ((44331.514 - elev_arr) / 11880.516) ** (1.0 / 0.1902632)
    return float(pressure) if pressure.ndim == 0 else pressure


def get_solar_position(times, lat, lon, elev=0, pressure=None, temp=12.0):
    r"""
    Calculates solar zenith angle ($Z$) and solar azimuth angle ($\phi$) using SPA [1]_ [2]_.
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
    pressure : float, optional
        Surface pressure. [hPa] If None, calculated from elevation.
        地表气压。[百帕]。若为 None，则根据海拔计算。
    temp : float, default 12.0
        Air temperature ($T$). [°C]
        气温 ($T$)。[摄氏度]

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
    
    # Calculate pressure if not provided (standard atmosphere)
    # 如果未提供气压，则根据标准大气压计算
    if pressure is None:
        # Use pvlib-equivalent alt2pres in Pa, then convert to hPa for SPA input.
        # 使用与 pvlib 等价的 alt2pres（Pa），再转换为 SPA 需要的 hPa。
        pressure = get_pressure_from_elevation(elev) / 100.0
    else:
        pressure_arr = np.asarray(pressure, dtype=float)
        if np.any(~np.isfinite(pressure_arr)) or np.any(pressure_arr <= 0):
            raise ValueError("pressure must be positive and finite. / pressure 必须为正且有限。")

        # Robust unit handling: SPA expects hPa. If values look like Pa, convert.
        # 稳健单位处理：SPA 需要 hPa。若数值看起来像 Pa，则自动转换。
        pressure_hpa = np.where(pressure_arr > 2000.0, pressure_arr / 100.0, pressure_arr)
        pressure = float(pressure_hpa) if pressure_hpa.ndim == 0 else pressure_hpa

    # Use fixed delta_t (69.1s for 2024) for simplicity
    # 为简便起见，使用固定的 delta_t（2024 年约为 69.1 秒）
    zenith, apparent_zenith, azimuth, _ = spa._solar_position(
        unixtime, lat, lon, elev, pressure=pressure, temp=temp, delta_t=69.1
    )
    
    solpos = pd.DataFrame({
        'zenith': zenith,
        'apparent_zenith': apparent_zenith,
        'azimuth': azimuth
    }, index=times)
    
    return solpos


def get_bni_extra(times):
    """
    Calculates extraterrestrial beam normal irradiance ($BNI_E$, $E_{0n}$) using Spencer (1971) [1]_.
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


def add_solpos_columns(df, station_code=None, lat=None, lon=None, elev=None):
    """
    Adds high-precision solar geometry and extraterrestrial irradiance columns to a DataFrame.
    向 DataFrame 添加高精度太阳几何和地外辐射列。

    Location can be given by BSRN station code or by explicit lat/lon/elev.
    位置可由 BSRN 站点代码指定，或由显式的 lat/lon/elev 指定。

    Parameters
    ----------
    df : pd.DataFrame
        Input data with pd.DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    station_code : str, optional
        BSRN station abbreviation. [e.g. 'QIQ'] Used if lat/lon/elev not provided.
        BSRN 站点缩写。[例如 'QIQ']。未提供 lat/lon/elev 时使用。
    lat : float, optional
        Latitude. [degrees] Required if station_code omitted.
        纬度。[度]。未提供 station_code 时必填。
    lon : float, optional
        Longitude. [degrees] Required if station_code omitted.
        经度。[度]。未提供 station_code 时必填。
    elev : float, optional
        Elevation. [m] Required if station_code omitted.
        海拔。[米]。未提供 station_code 时必填。

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'zenith', 'apparent_zenith', 'azimuth', 'bni_extra', and 'ghi_extra' columns.
        增加了 'zenith'、'apparent_zenith'、'azimuth'、'bni_extra' 和 'ghi_extra' 列的 DataFrame。

    Raises
    ------
    ValueError
        If ``df.index`` is not a :class:`~pandas.DatetimeIndex`. / 索引非 DatetimeIndex。
    ValueError
        If the median timestep is > 5 minutes (prevents non-linear geometric errors).
        若数据时间步长大于 5 分钟（防止非线性几何误差）。
    ValueError
        If neither a valid station_code nor complete (lat, lon, elev) is provided.
        若既未提供有效 station_code 也未提供完整的 (lat, lon, elev)。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex.")

    if len(df.index) > 1:
        # Check median time step size / 检查步长中位数
        median_dt = pd.Series(df.index).diff().median()
        if pd.notna(median_dt) and median_dt > pd.Timedelta(minutes=5):
            raise ValueError(
                f"Geometrical error: Calculating solar position on low-frequency data (timestep ≈ {median_dt}) "
                "introduces severe inaccuracies due to the non-linear diurnal path of the sun. "
                "Always compute solar position at high resolution (e.g., 1-minute) BEFORE averaging."
            )

    if lat is not None and lon is not None and elev is not None:
        pass  # use provided coordinates
    elif station_code is not None and station_code in BSRN_STATIONS:
        meta = BSRN_STATIONS[station_code]
        lat, lon, elev = meta["lat"], meta["lon"], meta["elev"]
    elif station_code is not None:
        raise ValueError(
            f"Station '{station_code}' not found in BSRN registry. "
            "For non-BSRN stations, provide 'lat', 'lon', and 'elev' explicitly. / "
            f"在 BSRN 注册表中未找到站点 '{station_code}'。非 BSRN 站点请显式提供 lat、lon、elev。"
        )
    else:
        raise ValueError(
            "Insufficient metadata. Provide a valid BSRN 'station_code' or "
            "explicit 'lat', 'lon', and 'elev'. / "
            "元数据不足。请提供有效的 BSRN 站点代码或显式的 lat、lon、elev。"
        )
    
    solpos = get_solar_position(df.index, lat, lon, elev)
    df["zenith"] = solpos["zenith"]
    df["apparent_zenith"] = solpos["apparent_zenith"]
    df["azimuth"] = solpos["azimuth"]
    
    df["bni_extra"] = get_bni_extra(df.index)
    df["ghi_extra"] = get_ghi_extra(df.index, df["zenith"])
    
    return df

