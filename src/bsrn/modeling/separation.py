"""
Irradiance separation models (Erbs, Engerer2, etc.).
Estimates diffuse fraction and DHI/BNI from GHI.
辐照分离模型（Erbs、Engerer2 等）。从 GHI 估算散射分数与 DHI/BNI。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.physics import clearsky
from bsrn.constants import ENGERER2_PARAMS, YANG4_PARAMS


def _get_solar_and_kt(df, lat, lon, ghi_col="ghi"):
    """
    Get GHI, extraterrestrial GHI, zenith, mu0, k_t, and night mask for separation.
    为分离模型计算 GHI、地外 GHI、天顶角、mu0、k_t 与夜间掩码。

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    ghi_col : str, default "ghi"
        Column name for GHI. [W/m^2]
        GHI 的列名。[瓦/平方米]

    Returns
    -------
    ghi, ghi_extra, zenith, mu0, k_t, night : tuple
        Solar and clearness index components.
        太阳和晴朗指数分量。
    """
    times = df.index
    ghi = np.asarray(df[ghi_col], dtype=float)
    # Get solar position / 获取太阳位置
    solpos = geometry.get_solar_position(times, lat, lon)
    zenith = solpos["zenith"].values
    # extraterrestrial GHI / 地外水平辐照度
    ghi_extra = geometry.get_ghi_extra(times, zenith).values
    mu0 = np.maximum(np.cos(np.radians(zenith)), 0.0)
    night = zenith >= 90
    ghi_extra_safe = np.where(ghi_extra > 0, ghi_extra, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_t = ghi / ghi_extra_safe
    k_t = np.where(night, np.nan, k_t)
    return ghi, ghi_extra, zenith, mu0, k_t, night


def _engerer2_k_at_resolution(df, lat, lon, period_minutes, ghi_col="ghi",
                              ghi_clear_col=None, station_code=None):
    """
    Compute Engerer2 diffuse fraction at a given temporal resolution by resampling.
    通过重采样在给定的时间分辨率下计算 Engerer2 散射分数。

    Resamples the input to `period_minutes`, runs Engerer2 with the corresponding
    coefficient set, and maps the resulting k back to the original index (forward fill
    within each period). Use this when you need true period-averaged Engerer2 k
    (e.g. k_d,60min for Yang4) rather than native-resolution k with period coefficients.
    将输入重采样到 `period_minutes`，运行具有相应系数集的 Engerer2，并将生成的 k 映射回原始索引
    （在每个周期内向前填充）。当您需要真实的周期平均 Engerer2 k（例如 Yang4 的 k_d,60min），
    而不是具有周期系数的原始分辨率 k 时，请使用此方法。

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    period_minutes : int
        Resampling resolution. [minutes]
        重采样分辨率。[分钟]
    ghi_col : str, default "ghi"
        Column name for GHI. [W/m^2]
        GHI 的列名。[瓦/平方米]
    ghi_clear_col : str or None, default None
        Column name for clear-sky GHI. [W/m^2]
        晴空 GHI 的列名。[瓦/平方米]
    station_code : str or None, default None
        BSRN station abbreviation. [e.g., 'QIQ']
        BSRN 站点缩写。[例如 'QIQ']

    Returns
    -------
    k : np.ndarray
        Diffuse fraction k aligned to `df.index`. [unitless]
        与 `df.index` 对齐的散射分数 k。[无单位]
    """
    resample_rule = {
        1: "1min",
        5: "5min",
        10: "10min",
        15: "15min",
        30: "30min",
        60: "1h",
        1440: "1D",
    }
    if period_minutes not in resample_rule:
        raise ValueError(
            "period_minutes must be one of 1, 5, 10, 15, 30, 60, 1440."
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    rule = resample_rule[period_minutes]
    cols = [ghi_col]
    if ghi_clear_col and ghi_clear_col in df.columns:
        cols.append(ghi_clear_col)
    # Compute counts of non-NA values per resampling bin
    counts = df[cols].resample(rule).count()
    bin_size = df.resample(rule).size()
    # Keep only those periods where more than half the data points are present for ghi_col
    enough_data = counts[ghi_col] > (bin_size / 2)
    # Compute mean as usual, set mean to NaN where not enough data
    df_rs = df[cols].resample(rule).mean()
    df_rs.loc[~enough_data] = np.nan
    df_rs = df_rs.dropna(subset=[ghi_col])

    ghi_cs = (ghi_clear_col if (ghi_clear_col and ghi_clear_col in df_rs.columns)
              else None)
    out = engerer2_separation(
        df_rs, lat, lon, ghi_col=ghi_col, ghi_clear_col=ghi_cs,
        station_code=station_code, averaging_period=period_minutes
    )
    k_series = out["k"].reindex(df.index, method="ffill")
    return np.asarray(k_series, dtype=float)


def _k_to_dhi_bni(ghi, k, zenith):
    """
    Convert diffuse fraction k to DHI and BNI. Night (zenith >= 90) yields NaN.
    由散射分数 k 计算 DHI 与 BNI；夜间为 NaN。

    Parameters
    ----------
    ghi : numeric or Series
        Global horizontal irradiance. [W/m^2]
        水平总辐照度。[瓦/平方米]
    k : numeric or Series
        Diffuse fraction. [unitless]
        散射分数。[无单位]
    zenith : numeric or Series
        Solar zenith angle. [degrees]
        太阳天顶角。[度]

    Returns
    -------
    dhi, bni : tuple
        Diffuse horizontal and beam normal irradiance. [W/m^2]
        水平散射与法向直接辐照度。[瓦/平方米]
    """
    ghi = np.asarray(ghi, dtype=float)
    k = np.asarray(k, dtype=float)
    zenith = np.asarray(zenith, dtype=float)
    night = zenith >= 90
    mu0 = np.cos(np.radians(zenith))
    mu0 = np.where(mu0 > 0, mu0, np.nan)
    dhi = ghi * k
    with np.errstate(divide="ignore", invalid="ignore"):
        bni = (ghi - dhi) / mu0
    dhi = np.where(night, np.nan, dhi)
    bni = np.where(night, np.nan, bni)
    return dhi, bni

def _brl_daily_clearness_index(times, ghi, ghi_extra):
    """
    Daily clearness index K_t = sum(ghi over day) / sum(ghi_extra over day).
    计算日晴朗指数 K_t = 一个全天 ghi 的总和 / ghi_extra 总和。

    Parameters
    ----------
    times : DatetimeIndex
        Timestamps for calculation.
        计算对应的时间戳。
    ghi : numeric or Series
        Global horizontal irradiance. [W/m^2]
        水平总辐照度。[瓦/平方米]
    ghi_extra : numeric or Series
        Extraterrestrial horizontal irradiance. [W/m^2]
        地外水平辐照度。[瓦/平方米]

    Returns
    -------
    K_t : np.ndarray
        Daily clearness index. [unitless]
        日晴朗指数。[无单位]
    """
    idx = pd.DatetimeIndex(times)
    ghi_ser = pd.Series(ghi, index=idx)
    ghi_extra_ser = pd.Series(ghi_extra, index=idx)
    hourly_ghi = ghi_ser.resample("1h").mean()
    hourly_ghi_extra = ghi_extra_ser.resample("1h").mean()
    daily_ghi = hourly_ghi.groupby(hourly_ghi.index.date).sum()
    daily_ghi_extra = hourly_ghi_extra.groupby(hourly_ghi_extra.index.date).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        K_t_daily = daily_ghi / daily_ghi_extra
    date_to_Kt = dict(zip(daily_ghi.index, K_t_daily.values))
    dates = np.array([t.date() for t in times])
    K_t = np.array([date_to_Kt.get(d, np.nan) for d in dates])
    return K_t

def _brl_psi(k_t, night, dates):
    """
    Piecewise linear interpolation for BRL ψ parameter.
    BRL ψ 参数的分段线性插值。

    ψ = (k_{t-1}+k_{t+1})/2 for sunrise < t < sunset; at sunrise ψ=k_{t+1}, at sunset ψ=k_{t-1}.
    在日出 < t < 日落时 ψ = (k_{t-1}+k_{t+1})/2；在日出时 ψ = k_{t+1}，在日落时 ψ = k_{t-1}。

    Parameters
    ----------
    k_t : numeric or Series
        Clearness index. [unitless]
        晴朗指数。[无单位]
    night : array-like
        Night mask (True for night).
        夜间掩码（夜间为 True）。
    dates : array-like
        Dates corresponding to timestamps.
        时间戳对应的日期。

    Returns
    -------
    psi : np.ndarray
        BRL ψ parameter. [unitless]
        BRL ψ 参数。[无单位]
    """
    n = len(k_t)
    psi = np.full(n, np.nan, dtype=float)
    k_t_arr = np.asarray(k_t, dtype=float)
    k_t_pad = np.concatenate([[np.nan], k_t_arr, [np.nan]])
    daytime = ~night
    # Per-day first and last daytime indices (sunrise / sunset)
    sunrise_idx = {}
    sunset_idx = {}
    for i in range(n):
        if not daytime[i]:
            continue
        d = dates[i]
        if d not in sunrise_idx:
            sunrise_idx[d] = i
        sunset_idx[d] = i
    for i in range(n):
        if not daytime[i]:
            continue
        d = dates[i]
        k_prev = k_t_pad[i]
        k_next = k_t_pad[i + 2]
        if i == sunrise_idx.get(d, -1):
            psi[i] = k_next if np.isfinite(k_next) else np.nan
        elif i == sunset_idx.get(d, -2):
            psi[i] = k_prev if np.isfinite(k_prev) else np.nan
        else:
            both = np.array([k_prev, k_next])
            psi[i] = np.nanmean(both) if np.any(np.isfinite(both)) else np.nan
    return psi
    
def erbs_separation(df, lat, lon, ghi_col="ghi"):
    """
    Erbs irradiance separation: diffuse fraction $k$ from clearness index $k_t$, then DHI and BNI.
    Erbs 辐照分离：由晴朗指数 $k_t$ 得散射分数 $k$，再得 DHI 与 BNI。

    Piecewise formula (Erbs et al.):
    - $k_t \\leq 0.22$: $k = 1.0 - 0.09 k_t$
    - $0.22 < k_t \\leq 0.80$: $k = 0.9511 - 0.1604 k_t + 4.388 k_t^2 - 16.638 k_t^3 + 12.336 k_t^4$
    - $k_t > 0.80$: $k = 0.165$

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    ghi_col : str, default "ghi"
        Column name for GHI. [W/m^2]
        GHI 的列名。[瓦/平方米]

    Returns
    -------
    out : pd.DataFrame
        Copy of `df` with added columns `k`, `dhi`, `bni`. [unitless, W/m^2, W/m^2]
        增加了 `k`、`dhi`、`bni` 列的 `df` 副本。[无单位, 瓦/平方米, 瓦/平方米]

    References
    ----------
    .. [1] Erbs, D. G., Klein, S. A., & Duffie, J. A. (1982). Estimation of 
       the diffuse radiation fraction for hourly, daily and monthly-average 
       global radiation. Solar Energy, 28(4), 293-302.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    ghi, ghi_extra, zenith, mu0, k_t, night = _get_solar_and_kt(df, lat, lon, ghi_col)

    # Initialize k array / 初始化 k 数组
    k = np.full_like(k_t, np.nan, dtype=float)
    # Clearness index conditions / 晴朗指数条件
    low = np.isfinite(k_t) & (k_t <= 0.22)
    mid = np.isfinite(k_t) & (k_t > 0.22) & (k_t <= 0.80)
    high = np.isfinite(k_t) & (k_t > 0.80)

    k[low] = 1.0 - 0.09 * k_t[low]
    kt = k_t[mid]
    k[mid] = (
        0.9511
        - 0.1604 * kt
        + 4.388 * (kt ** 2)
        - 16.638 * (kt ** 3)
        + 12.336 * (kt ** 4)
    )
    k[high] = 0.165

    k = np.clip(k, 0.0, 1.0)
    k = np.where(night, np.nan, k)

    dhi, bni = _k_to_dhi_bni(ghi, k, zenith)

    out = df.copy()
    out["k"] = k
    out["dhi"] = dhi
    out["bni"] = bni
    return out

def engerer2_separation(df, lat, lon, ghi_col="ghi", ghi_clear_col=None,
                        station_code=None, averaging_period=1):
    """
    Engerer2 irradiance separation: estimate diffuse fraction ($k$), DHI and BNI from GHI.
    Engerer2 辐照分离：由 GHI 估算散射分数 ($k$)、DHI 与 BNI。

    Uses the re-parameterized Engerer2 model (Bright & Engerer 2019).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    ghi_col : str, default "ghi"
        Column name for GHI. [W/m^2]
        GHI 的列名。[瓦/平方米]
    ghi_clear_col : str or None, default None
        Column name for clear-sky GHI. [W/m^2]
        晴空 GHI 的列名。[瓦/平方米]
    station_code : str or None, default None
        BSRN station abbreviation. [e.g., 'QIQ']
        BSRN 站点缩写。[例如 'QIQ']
    averaging_period : int, default 1
        Coefficient set for native resolution. [minutes]
        对应原始分辨率的系数集。[分钟]

    Returns
    -------
    out : pd.DataFrame
        Copy of `df` with added columns: `k`, `dhi`, `bni`. [unitless, W/m^2, W/m^2]
        增加了 `k`、`dhi`、`bni` 列的 `df` 副本。[无单位, 瓦/平方米, 瓦/平方米]

    References
    ----------
    .. [1] Bright, J. M., & Engerer, N. A. (2019). Engerer2: Global 
       re-parameterisation, update, and validation of an irradiance 
       separation model at different temporal resolutions. Journal of 
       Renewable and Sustainable Energy, 11(3), 033701.
    .. [2] Engerer, N. A. (2015). Minute resolution estimates of the 
       diffuse fraction of global irradiance for southeastern Australia. 
       Solar Energy, 116, 215-237.
    """
    if averaging_period not in ENGERER2_PARAMS:
        raise ValueError(
            "averaging_period must be one of 1, 5, 10, 15, 30, 60, 1440 (minutes)."
        )

    times = df.index
    if not isinstance(times, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    ghi = np.asarray(df[ghi_col], dtype=float)
    solpos = geometry.get_solar_position(times, lat, lon)
    zenith = solpos["zenith"].values
    ghi_extra = geometry.get_ghi_extra(times, zenith).values

    doy = times.dayofyear.values
    decimal_hour = (
        times.hour.values
        + times.minute.values / 60.0
        + times.second.values / 3600.0
    )
    beta_eot = (360.0 / 365.242) * (doy - 1)
    eot = (
        0.258 * np.cos(np.radians(beta_eot))
        - 7.416 * np.sin(np.radians(beta_eot))
        - 3.648 * np.cos(np.radians(2 * beta_eot))
        - 9.228 * np.sin(np.radians(2 * beta_eot))
    )
    lsn = 12 - lon / 15.0 - eot / 60.0
    hour_angle = (decimal_hour - lsn) * 15.0
    hour_angle = np.where(hour_angle >= 180, hour_angle - 360, hour_angle)
    hour_angle = np.where(hour_angle <= -180, hour_angle + 360, hour_angle)
    ast = hour_angle / 15.0 + 12.0
    ast = np.where(ast < 0, np.abs(ast), ast)

    if ghi_clear_col is not None and ghi_clear_col in df.columns:
        ghi_clear = np.asarray(df[ghi_clear_col], dtype=float)
    elif station_code is not None:
        df_cs = clearsky.add_clearsky_columns(df[[ghi_col]].copy(), station_code)
        ghi_clear = np.asarray(df_cs["ghi_clear"], dtype=float)
    else:
        ghi_clear = clearsky.threlkeld_jordan_model(zenith, doy)

    ghi_extra_safe = np.where(ghi_extra > 0, ghi_extra, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_t = ghi / ghi_extra_safe
        ktc = ghi_clear / ghi_extra_safe
    dktc = ktc - k_t
    cloud_enh = np.where(ghi - ghi_clear > 0.015, ghi - ghi_clear, 0.0)
    k_de = np.where(ghi > 0, cloud_enh / ghi, 0.0)

    night = zenith >= 90
    k_t = np.where(night, np.nan, k_t)
    ktc = np.where(night, np.nan, ktc)
    dktc = np.where(night, np.nan, dktc)
    ast = np.where(night, np.nan, ast)
    k_de = np.where(night, np.nan, k_de)

    # Engerer2 logistic formula / Engerer2 逻辑公式
    c, b0, b1, b2, b3, b4, b5 = ENGERER2_PARAMS[averaging_period]
    with np.errstate(invalid="ignore"):
        k = c + (1 - c) / (1 + np.exp(
            b0 + b1 * k_t + b2 * ast + b3 * zenith + b4 * dktc
        )) + b5 * k_de
    k = np.clip(k, 0.0, 1.0)
    k = np.where(night, np.nan, k)

    dhi = ghi * k
    mu0 = np.maximum(np.cos(np.radians(zenith)), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bni = np.where(~night, (ghi - dhi) / mu0, np.nan)
    dhi = np.where(night, np.nan, dhi)

    out = df.copy()
    out["k"] = k
    out["dhi"] = dhi
    out["bni"] = bni
    return out

def brl_separation(df, lat, lon, ghi_col="ghi"):
    """
    BRL irradiance separation: diffuse fraction d from logistic function of k_t, AST, alpha, K_t, psi.
    BRL 辐照分离：由 k_t、AST、alpha、K_t、psi 的逻辑回归函数得散射分数 d。

    d = 1 / (1 + exp(-5.38 + 6.63*k_t + 0.006*AST - 0.007*alpha + 1.75*K_t + 1.31*psi)).
    psi at sunrise = k_{t+1}, at sunset = k_{t-1}, else (k_{t-1}+k_{t+1})/2. K_t = daily clearness index.
    在日出时 psi = k_{t+1}，在日落时 psi = k_{t-1}，否则为 (k_{t-1}+k_{t+1})/2。K_t = 日晴朗指数。

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    ghi_col : str, default "ghi"
        Column name for GHI. [W/m^2]
        GHI 的列名。[瓦/平方米]

    Returns
    -------
    out : pd.DataFrame
        Copy of `df` with added columns `k`, `dhi`, `bni`. [unitless, W/m^2, W/m^2]
        增加了 `k`、`dhi`、`bni` 列的 `df` 副本。[无单位, 瓦/平方米, 瓦/平方米]

    References
    ----------
    .. [1] Ridley, B., Boland, J., & Lauret, P. (2010). Modelling of 
       diffuse solar fraction with multiple predictors. Renewable 
       Energy, 35(2), 478-483.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    times = df.index
    ghi, ghi_extra, zenith, mu0, k_t, night = _get_solar_and_kt(df, lat, lon, ghi_col)

    # Daily clearness index K_t (Eq. 7) / 日晴朗指数 K_t (公式 7)
    K_t = _brl_daily_clearness_index(times, ghi, ghi_extra)

    # Apparent solar time AST (hours) / 地面太阳时 AST (小时)
    doy = times.dayofyear.values
    decimal_hour = (
        times.hour.values
        + times.minute.values / 60.0
        + times.second.values / 3600.0
    )
    beta_eot = (360.0 / 365.242) * (doy - 1)
    eot = (
        0.258 * np.cos(np.radians(beta_eot))
        - 7.416 * np.sin(np.radians(beta_eot))
        - 3.648 * np.cos(np.radians(2 * beta_eot))
        - 9.228 * np.sin(np.radians(2 * beta_eot))
    )
    lsn = 12 - lon / 15.0 - eot / 60.0
    hour_angle = (decimal_hour - lsn) * 15.0
    hour_angle = np.where(hour_angle >= 180, hour_angle - 360, hour_angle)
    hour_angle = np.where(hour_angle <= -180, hour_angle + 360, hour_angle)
    ast = hour_angle / 15.0 + 12.0
    ast = np.where(ast < 0, np.abs(ast), ast)

    # Solar altitude alpha (degrees) = 90 - zenith / 太阳高度角 alpha (度) = 90 - 天顶角
    alpha = 90.0 - zenith

    # psi: piecewise from k_t at adjacent timesteps / psi: 来自相邻时间步长的 k_t 的分段函数
    dates = np.array([t.date() for t in times])
    psi = _brl_psi(k_t, night, dates)

    # d = 1 / (1 + exp(...)) / d 逻辑回归公式
    exponent = -5.38 + 6.63 * k_t + 0.006 * ast - 0.007 * alpha + 1.75 * K_t + 1.31 * psi
    with np.errstate(invalid="ignore", over="ignore"):
        k = 1.0 / (1.0 + np.exp(-exponent))
    k = np.clip(k, 0.0, 1.0)
    k = np.where(night, np.nan, k)

    dhi, bni = _k_to_dhi_bni(ghi, k, zenith)

    out = df.copy()
    out["k"] = k
    out["dhi"] = dhi
    out["bni"] = bni
    return out

def yang4_separation(df, lat, lon, ghi_col="ghi", ghi_clear_col=None,
                     station_code=None):
    """
    Yang4 irradiance separation: diffuse fraction k_d from k_t, AST, Z, Δk_tc, k_de, and Engerer2 60-min k.
    k_d^YANG4 = C + (1-C)/(1 + exp(β0 + β1*k_t + β2*AST + β3*Z + β4*Δk_tc + β6*k_d,60min^ENGERER2)) + β5*k_de.
    Uses YANG2 coefficient set (TABLE III) from 1-min SURFRAD data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex.
        包含 DatetimeIndex 的输入数据。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    ghi_col : str, default "ghi"
        Column name for GHI. [W/m^2]
        GHI 的列名。[瓦/平方米]
    ghi_clear_col : str or None, default None
        Column name for clear-sky GHI. [W/m^2]
        晴空 GHI 的列名。[瓦/平方米]
    station_code : str or None, default None
        BSRN station abbreviation. [e.g., 'QIQ']
        BSRN 站点缩写。[例如 'QIQ']

    Returns
    -------
    out : pd.DataFrame
        Copy of `df` with added columns: `k`, `dhi`, `bni`. [unitless, W/m^2, W/m^2]
        增加了 `k`、`dhi`、`bni` 列的 `df` 副本。[无单位, 瓦/平方米, 瓦/平方米]

    References
    ----------
    .. [1] Yang, D. (2021). Temporal-resolution cascade model for 
       separation of 1-min beam and diffuse irradiance. Journal of 
       Renewable and Sustainable Energy, 13(5), 053703.
    .. [2] Yang, D., & Boland, J. (2019). Satellite-augmented diffuse 
       solar radiation separation models. Journal of Renewable and 
       Sustainable Energy, 11(2), 023704.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    # k_d,60min^ENGERER2: true hourly-averaged Engerer2 k
    # k_d,60min^ENGERER2: 真实的小时平均 Engerer2 k
    k_engerer2_60 = _engerer2_k_at_resolution(
        df, lat, lon, 60, ghi_col=ghi_col,
        ghi_clear_col=ghi_clear_col, station_code=station_code
    )

    times = df.index
    ghi = np.asarray(df[ghi_col], dtype=float)
    solpos = geometry.get_solar_position(times, lat, lon)
    zenith = solpos["zenith"].values
    ghi_extra = geometry.get_ghi_extra(times, zenith).values
    doy = times.dayofyear.values
    decimal_hour = (
        times.hour.values
        + times.minute.values / 60.0
        + times.second.values / 3600.0
    )
    beta_eot = (360.0 / 365.242) * (doy - 1)
    eot = (
        0.258 * np.cos(np.radians(beta_eot))
        - 7.416 * np.sin(np.radians(beta_eot))
        - 3.648 * np.cos(np.radians(2 * beta_eot))
        - 9.228 * np.sin(np.radians(2 * beta_eot))
    )
    lsn = 12 - lon / 15.0 - eot / 60.0
    hour_angle = (decimal_hour - lsn) * 15.0
    hour_angle = np.where(hour_angle >= 180, hour_angle - 360, hour_angle)
    hour_angle = np.where(hour_angle <= -180, hour_angle + 360, hour_angle)
    ast = hour_angle / 15.0 + 12.0
    ast = np.where(ast < 0, np.abs(ast), ast)

    if ghi_clear_col is not None and ghi_clear_col in df.columns:
        ghi_clear = np.asarray(df[ghi_clear_col], dtype=float)
    elif station_code is not None:
        df_cs = clearsky.add_clearsky_columns(df[[ghi_col]].copy(), station_code)
        ghi_clear = np.asarray(df_cs["ghi_clear"], dtype=float)
    else:
        ghi_clear = clearsky.threlkeld_jordan_model(zenith, doy)

    ghi_extra_safe = np.where(ghi_extra > 0, ghi_extra, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_t = ghi / ghi_extra_safe
        ktc = ghi_clear / ghi_extra_safe
    dktc = ktc - k_t
    cloud_enh = np.where(ghi - ghi_clear > 0.015, ghi - ghi_clear, 0.0)
    k_de = np.where(ghi > 0, cloud_enh / ghi, 0.0)

    night = zenith >= 90
    k_t = np.where(night, np.nan, k_t)
    dktc = np.where(night, np.nan, dktc)
    ast = np.where(night, np.nan, ast)
    k_de = np.where(night, np.nan, k_de)
    k_engerer2_60 = np.where(night, np.nan, k_engerer2_60)

    # Yang4 logistic formula / Yang4 逻辑公式
    C, b0, b1, b2, b3, b4, b5, b6 = YANG4_PARAMS
    exponent = b0 + b1 * k_t + b2 * ast + b3 * zenith + b4 * dktc + b6 * k_engerer2_60
    with np.errstate(invalid="ignore", over="ignore"):
        k = C + (1 - C) / (1 + np.exp(exponent)) + b5 * k_de
    k = np.clip(k, 0.0, 1.0)
    k = np.where(night, np.nan, k)

    dhi, bni = _k_to_dhi_bni(ghi, k, zenith)

    out = df.copy()
    out["k"] = k
    out["dhi"] = dhi
    out["bni"] = bni
    return out
