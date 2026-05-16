"""
Solar geometry and extraterrestrial irradiance helpers.

Uses the internal SPA implementation in :mod:`bsrn.physics.spa` for
:func:`get_solar_position`; public callers use this module only.
"""

import pandas as pd
import numpy as np
from bsrn.physics import spa
from bsrn.constants import BSRN_STATIONS, GEO_SATELLITE_LAT_DEG, GEO_SATELLITE_LON_DEG, GEO_SATELLITE_DISK_RADIUS_DEG


def get_pressure_from_elevation(elev):
    """
    Calculates standard-atmosphere pressure from elevation using pvlib-equivalent formula [1]_.

    Parameters
    ----------
    elev : numeric
        Elevation above sea level. [m]

    Returns
    -------
    pressure : numeric
        Surface pressure. [Pa]

    Raises
    ------
    ValueError
        If *elev* is non-finite, or ``elev >= 44331.514`` m (outside the valid
        standard-atmosphere range for this formula).

    References
    ----------
    .. [1] Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python:
       A python package for modeling solar energy systems. Journal of Open Source Software,
       3(29), 884.
    """
    elev_arr = np.asarray(elev, dtype=float)
    if np.any(~np.isfinite(elev_arr)):
        raise ValueError("elev must be finite.")

    # Keep real-valued output and fail early for out-of-range altitudes.
    if np.any(elev_arr >= 44331.514):
        raise ValueError("elev must be < 44331.514 m for standard-atmosphere formula.")

    pressure = 100.0 * ((44331.514 - elev_arr) / 11880.516) ** (1.0 / 0.1902632)
    return float(pressure) if pressure.ndim == 0 else pressure


def get_solar_position(times, lat, lon, elev=0, pressure=None, temp=12.0):
    r"""
    Calculates solar zenith angle ($Z$) and solar azimuth angle ($\phi$) using SPA [1]_ [2]_.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
    lat : float
        Latitude. [degrees]
    lon : float
        Longitude. [degrees]
    elev : float, default 0
        Elevation. [m]
    pressure : float, optional
        Surface pressure. [hPa] If None, calculated from elevation.
    temp : float, default 12.0
        Air temperature ($T$). [°C]

    Returns
    -------
    solpos : pd.DataFrame
        DataFrame with columns 'zenith' ($Z$), 'apparent_zenith', 'azimuth' ($\phi$). [degrees]

    Raises
    ------
    ValueError
        If *pressure* is provided and is non-positive or non-finite, or if
        :func:`get_pressure_from_elevation` rejects *elev* when *pressure* is omitted.

    References
    ----------
    .. [1] Holmgren, W. F., Hansen, C. W., & Mikofski, M. A. (2018). pvlib python: 
       A python package for modeling solar energy systems. Journal of Open 
       Source Software, 3(29), 884.
    .. [2] Anderson, K. S., et al. (2023). pvlib python: 2023 project update. 
       Journal of Open Source Software, 8(92), 5994.
    """
    # Unix time [s]. Epoch subtraction is robust to DatetimeIndex resolution
    # (e.g. datetime64[ns] vs datetime64[us]); ``.view(np.int64) / 1e9`` is not.
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    unixtime = ((times - _epoch) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    
    # Calculate pressure if not provided (standard atmosphere)
    if pressure is None:
        # Use pvlib-equivalent alt2pres in Pa, then convert to hPa for SPA input.
        pressure = get_pressure_from_elevation(elev) / 100.0
    else:
        pressure_arr = np.asarray(pressure, dtype=float)
        if np.any(~np.isfinite(pressure_arr)) or np.any(pressure_arr <= 0):
            raise ValueError("pressure must be positive and finite.")

        # Robust unit handling: SPA expects hPa. If values look like Pa, convert.
        pressure_hpa = np.where(pressure_arr > 2000.0, pressure_arr / 100.0, pressure_arr)
        pressure = float(pressure_hpa) if pressure_hpa.ndim == 0 else pressure_hpa

    # Use fixed delta_t (69.1s for 2024) for simplicity
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

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.

    Returns
    -------
    bni_extra : pd.Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]

    References
    ----------
    .. [1] Spencer, J. W. (1971). Fourier series representation of the sun. 
       Search, 2, 172.
    """
    # Day angle (radians) using Spencer (1971): (2*pi/365)*(doy - 1)
    day_of_year = times.dayofyear
    b = (2.0 * np.pi / 365.0) * (day_of_year - 1.0)
    
    # Eccentricity correction
    # E0 = Isc * (1.00011 + 0.034221*cos(B) + 0.001280*sin(B) + 0.000719*cos(2B) + 0.000077*sin(2B))
    r_r0_sq = (1.00011 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b) + 
               0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))
    
    # Using the project defined solar constant
    from bsrn.constants import solar_constant
    bni_extra = solar_constant * r_r0_sq
    
    return pd.Series(bni_extra, index=times)


def get_ghi_extra(times, zenith):
    """
    Calculates extraterrestrial horizontal irradiance ($GHI_E$, $E_0$).

    Parameters
    ----------
    times : pd.DatetimeIndex
        Times for calculation.
    zenith : numeric or Series
        Solar zenith angle ($Z$). [degrees]

    Returns
    -------
    ghi_extra : pd.Series
        Extraterrestrial horizontal irradiance ($E_0$). [W/m^2]
    """
    bni_extra = get_bni_extra(times)
    mu0 = np.cos(np.radians(zenith))
    return bni_extra * np.maximum(mu0, 0)


def add_solpos_columns(df, station_code=None, lat=None, lon=None, elev=None):
    """
    Adds high-precision solar geometry and extraterrestrial irradiance columns to a DataFrame.

    Location can be given by BSRN station code or by explicit lat/lon/elev.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with pd.DatetimeIndex.
    station_code : str, optional
        BSRN station abbreviation. [e.g. 'QIQ'] Used if lat/lon/elev not provided.
    lat : float, optional
        Latitude. [degrees] Required if station_code omitted.
    lon : float, optional
        Longitude. [degrees] Required if station_code omitted.
    elev : float, optional
        Elevation. [m] Required if station_code omitted.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added 'zenith', 'apparent_zenith', 'azimuth', 'bni_extra', and 'ghi_extra' columns.

    Raises
    ------
    ValueError
        If ``df.index`` is not a :class:`~pandas.DatetimeIndex`.
    ValueError
        If the median timestep is > 5 minutes (prevents non-linear geometric errors).
    ValueError
        If neither a valid station_code nor complete (lat, lon, elev) is provided.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex.")

    if len(df.index) > 1:
        # Check median time step size
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
            "For non-BSRN stations, provide 'lat', 'lon', and 'elev' explicitly."
        )
    else:
        raise ValueError(
            "Insufficient metadata. Provide a valid BSRN 'station_code' or "
            "explicit 'lat', 'lon', and 'elev'."
        )
    
    solpos = get_solar_position(df.index, lat, lon, elev)
    df["zenith"] = solpos["zenith"].round(3)
    df["apparent_zenith"] = solpos["apparent_zenith"].round(3)
    df["azimuth"] = solpos["azimuth"].round(3)
    
    df["bni_extra"] = get_bni_extra(df.index).round(3)
    df["ghi_extra"] = get_ghi_extra(df.index, df["zenith"]).round(3)
    return df


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def _central_angle_deg(lat1, lon1, lat2, lon2):
    """
    Great-circle central angle between two surface points using the Haversine formula. [degrees]

    Parameters
    ----------
    lat1 : float or numeric
        Latitude of the first point. [degrees]
    lon1 : float or numeric
        Longitude of the first point. [degrees]
    lat2 : float or numeric
        Latitude of the second point. [degrees]
    lon2 : float or numeric
        Longitude of the second point. [degrees]

    Returns
    -------
    angle : float or numeric
        Central angle in degrees.
    """
    # Convert degrees to radians
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    # Haversine half-chord squared
    h = (
        np.sin(dlat / 2) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    )
    # Return central angle [degrees]
    return 2 * np.degrees(np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def in_satellite_disk(lat, lon, sat_key):
    """
    True if (lat, lon) falls within the Earth-disk radius (approx. 60 deg) of the satellite.

    Parameters
    ----------
    lat : float or numeric
        Latitude. [degrees]
    lon : float or numeric
        Longitude. [degrees]
    sat_key : str
        Satellite key in ``bsrn.constants.GEO_SATELLITES`` (e.g., 'Himawari', 'MSG').

    Returns
    -------
    covered : bool or boolean array
        True if covered, False otherwise.

    Raises
    ------
    KeyError
        If *sat_key* is missing from the satellite longitude table
        (see ``GEO_SATELLITE_LON_DEG``).
    """
    # Angular distance to the sub-satellite point
    angle = _central_angle_deg(
        lat, lon,
        GEO_SATELLITE_LAT_DEG, GEO_SATELLITE_LON_DEG[sat_key],
    )
    # Check if within the 60° reliability limit
    return angle <= GEO_SATELLITE_DISK_RADIUS_DEG
