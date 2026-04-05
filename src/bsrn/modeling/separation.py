"""
Irradiance separation models (Erbs, Engerer2, etc.).
Estimates diffuse fraction and DHI/BNI from GHI.
辐照分离模型（Erbs、Engerer2 等）。从 GHI 估算散射分数与 DHI/BNI。
"""

import numpy as np
import pandas as pd
from bsrn.physics import geometry
from bsrn.constants import ENGERER2_PARAMS, YANG4_PARAMS
from bsrn.utils.calculations import calc_kt


def _get_solar_and_kt(times, ghi, lat, lon, elev=0, min_mu0=0.065,
                      max_clearness_index=1.0):
    """
    Get ghi_extra, zenith, mu0, kt, and night mask for separation.
    为分离模型计算 ghi_extra、天顶角、mu0、kt 与夜间掩码。

    Uses pvlib-style clearness index: ghi_extra = bni_extra * max(mu0, min_mu0),
    kt from calc_kt, clamped and set to NaN at night.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps (same length as ghi).
        时间戳（与 ghi 等长）。
    ghi : array-like
        Global horizontal irradiance. [W/m^2]
        水平总辐照度。[瓦/平方米]
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    elev : float, default 0
        Site elevation. [m] Used for topocentric solar position (matches pvlib when same elev).
        站点海拔。[米]。用于站心太阳位置（与 pvlib 使用相同 elev 时一致）。
    min_mu0 : float, default 0.065
        Minimum cosine of solar zenith for ghi_extra and kt (equiv. ~86.3 deg).
        ghi_extra 与 kt 中余弦天顶角的最小值（相当于 ~86.3 度）。
    max_clearness_index : float, default 1.0
        Upper clamp for kt.
        kt 的上限。

    Returns
    -------
    ghi, ghi_extra, zenith, mu0, kt, night : tuple
        Solar and clearness index components.
        太阳和晴朗指数分量。

    Raises
    ------
    ValueError
        If ``times`` is not a :class:`~pandas.DatetimeIndex` or ``ghi`` length mismatches.
        ``times`` 非 DatetimeIndex 或 ``ghi`` 长度不一致时。
    """

    # Check if times is a pd.DatetimeIndex.
    # 检查 times 是否为 pd.DatetimeIndex。
    if not isinstance(times, pd.DatetimeIndex):
        raise ValueError("times must be a pd.DatetimeIndex.")

    ghi = np.asarray(ghi, dtype=float)
    if len(ghi) != len(times):
        raise ValueError("ghi must have the same length as times.")

    # Get the solar position. / 获取太阳位置。
    solpos = geometry.get_solar_position(times, lat, lon, elev)
    zenith = solpos["zenith"].values
    mu0 = np.maximum(np.cos(np.radians(zenith)), 0.0)

    # Get the extraterrestrial BNI. / 获取地外法向辐照度。
    bni_extra = np.asarray(geometry.get_bni_extra(times), dtype=float)
    ghi_extra = bni_extra * np.maximum(mu0, min_mu0)

    # Get the night mask. / 获取夜间掩码。
    night = zenith >= 90

    # Get the clearness index. / 获取晴朗指数。
    # Keep calc_kt output as-is; nighttime separation (k, dhi, bni) is set to 0 in each model.
    kt = calc_kt(ghi, zenith, bni_extra, min_mu0=min_mu0,
                  max_clearness_index=max_clearness_index)
    return ghi, ghi_extra, zenith, mu0, kt, night

def _k_to_dhi_bni(ghi, k, zenith, max_zenith=87.0, force_nan=None):
    """
    Convert diffuse fraction $k$ to DHI and BNI with pvlib-style edge handling.
    由散射分数 $k$ 计算 DHI 与 BNI，采用 pvlib 风格的边界处理。

    DHI = $k \\cdot G_h$, BNI = $(G_h - \\text{DHI}) / \\mu_0$. Where
    zenith > max_zenith, GHI < 0, BNI < 0, or $\\mu_0 \\le 0$, sets BNI = 0 and
    DHI = GHI to preserve closure. Optionally force DHI/BNI to NaN where
    force_nan is True (e.g. under-sampled days in BRL).

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance ($G_h$). [W/m^2]
        水平总辐照度。[瓦/平方米]
    k : array-like
        Diffuse fraction ($k$). [unitless]
        散射分数。[无单位]
    zenith : array-like
        Solar zenith angle ($Z$). [degrees]
        太阳天顶角。[度]
    max_zenith : float, default 87.0
        Maximum zenith for valid BNI; beyond this BNI = 0, DHI = GHI. [degrees]
        BNI 有效的天顶角上限；超过则 BNI=0、DHI=GHI。[度]
    force_nan : array-like or None, default None
        If provided, DHI and BNI are set to NaN where force_nan is True.
        若提供，则在 force_nan 为 True 处将 DHI、BNI 设为 NaN。

    Returns
    -------
    dhi, bni : tuple of np.ndarray
        Diffuse horizontal and beam normal irradiance. [W/m^2]
        水平散射与法向直接辐照度。[瓦/平方米]
    """

    ghi = np.asarray(ghi, dtype=float)
    k = np.asarray(k, dtype=float)
    zenith = np.asarray(zenith, dtype=float)
    mu0 = np.maximum(np.cos(np.radians(zenith)), 0.0)
    dhi = ghi * k
    with np.errstate(divide="ignore", invalid="ignore"):
        bni = (ghi - dhi) / mu0
    bad_values = (zenith > max_zenith) | (ghi < 0) | (bni < 0) | (mu0 <= 0)
    bni = np.where(bad_values, 0.0, bni)
    dhi = np.where(bad_values, ghi, dhi)
    if force_nan is not None:
        force_nan = np.asarray(force_nan, dtype=bool)
        dhi = np.where(force_nan, np.nan, dhi)
        bni = np.where(force_nan, np.nan, bni)
    return dhi, bni
    
def _apparent_solar_time(times, lon):
    """
    Calculate apparent solar time (AST, hours) from timestamps and longitude.
    根据时间戳 and 经度计算地面太阳时 AST（小时）。

    Uses the equation of time formulation from Ridley et al. (2010), consistent
    with the BRL, Engerer2 and Yang4 models.
    使用与 BRL、Engerer2 与 Yang4 模型一致的时间方程形式（Ridley 等，2010）。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps for calculation.
        计算对应的时间戳。
    lon : float
        Longitude. [degrees]
        经度。[度]

    Returns
    -------
    ast : np.ndarray
        Apparent solar time in hours. [h]
        地面太阳时（小时）。[小时]
    """

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
    return ast

def _brl_daily_clearness_index(times, ghi, ghi_extra, night):
    """
    Daily clearness index K_t = sum(ghi over day) / sum(ghi_extra over day).
    计算日晴朗指数 K_t = 一个全天 ghi 的总和 / ghi_extra 总和。

    A daily K_t is only computed when more than half of the daytime hourly
    values (ghi_extra > 0) have non-NaN GHI; otherwise K_t for that day is NaN.
    只有当白天小时数据（ghi_extra > 0）中超过一半具有非 NaN 的 GHI 时，才计算该日的 K_t，
    否则该日的 K_t 记为 NaN。

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
    night : array-like
        Night mask at the native resolution (True for night).
        原始时间分辨率下的夜间掩码（夜间为 True）。

    Returns
    -------
    Kt : np.ndarray
        Daily clearness index ($K_t$). [unitless]
        日晴朗指数 ($K_t$)。[无单位]
    """

    idx = pd.DatetimeIndex(times)
    ghi_ser = pd.Series(ghi, index=idx)
    ghi_extra_ser = pd.Series(ghi_extra, index=idx)
    night_ser = pd.Series(night, index=idx)

    # Resample to hourly means (GHI, GHI_extra) and any-night mask / 重采样到小时均值和“是否夜间”掩码
    hourly_ghi = ghi_ser.resample("1h").mean()
    hourly_ghi_extra = ghi_extra_ser.resample("1h").mean()
    hourly_night = night_ser.resample("1h").max().astype(bool)

    # Define daytime as hours with at least one non-night sample / 若该小时内存在非夜间样本，则视为白天
    is_daytime = ~hourly_night
    has_data = is_daytime & hourly_ghi.notna()

    # Per-day sums and counts restricted to daytime / 白天内的逐日求和和计数
    dates = hourly_ghi.index.date
    daily_ghi = hourly_ghi.where(is_daytime).groupby(dates).sum()
    daily_ghi_extra = hourly_ghi_extra.where(is_daytime).groupby(dates).sum()
    daily_count_daytime = is_daytime.groupby(dates).sum()
    daily_count_valid = has_data.groupby(dates).sum()

    # Only compute K_t when > half of daytime hours have valid data / 仅当有效白天小时数超过一半时计算 K_t
    enough_data = daily_count_valid > (daily_count_daytime / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kt_day = daily_ghi / daily_ghi_extra
    Kt_day = Kt_day.where(enough_data, np.nan)

    date_to_Kt = dict(zip(Kt_day.index, Kt_day.values))
    dates = np.array([t.date() for t in times])
    Kt = np.array([date_to_Kt.get(d, np.nan) for d in dates])
    return Kt

def _brl_psi(kt, night, dates):
    """
    Piecewise linear interpolation for BRL ψ parameter.
    BRL ψ 参数的分段线性插值。

    ψ = (k_{t-1}+k_{t+1})/2 for sunrise < t < sunset; at sunrise ψ=k_{t+1}, at sunset ψ=k_{t-1}.
    在日出 < t < 日落时 ψ = (k_{t-1}+k_{t+1})/2；在日出时 ψ = k_{t+1}，在日落时 ψ = k_{t-1}。

    Parameters
    ----------
    kt : array-like
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

    n = len(kt)
    psi = np.full(n, np.nan, dtype=float)
    kt_arr = np.asarray(kt, dtype=float)
    kt_pad = np.concatenate([[np.nan], kt_arr, [np.nan]])
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
        k_prev = kt_pad[i]
        k_next = kt_pad[i + 2]
        if i == sunrise_idx.get(d, -1):
            psi[i] = k_next if np.isfinite(k_next) else np.nan
        elif i == sunset_idx.get(d, -2):
            psi[i] = k_prev if np.isfinite(k_prev) else np.nan
        else:
            both = np.array([k_prev, k_next])
            psi[i] = np.nanmean(both) if np.any(np.isfinite(both)) else np.nan
    return psi


def _engerer2_k_at_resolution(df, lat, lon, period_minutes, ghi_col="ghi",
                              ghi_clear_col="ghi_clear"):
    """
    Compute Engerer2 diffuse fraction at a given temporal resolution by resampling.
    通过重采样在给定的时间分辨率下计算 Engerer2 散射分数。

    Requires a clear-sky GHI column in df (caller must provide it).
    要求 df 中包含晴空 GHI 列（由调用方提供）。

    Resamples the input to `period_minutes` using right-closed bins (e.g. (10:00, 11:00]
    for the hour labeled 11:00), runs Engerer2 with the corresponding coefficient set,
    and maps the resulting k back to the original index using first-future (backward fill):
    each 1-min time t in (10:00, 11:00] gets the hourly k at 11:00. Use this when you
    need true period-averaged Engerer2 k (e.g. k_60min for Yang4).

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex and a clear-sky GHI column.
        包含 DatetimeIndex 与晴空 GHI 列的输入数据。
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
    ghi_clear_col : str, default "ghi_clear"
        Column name for clear-sky GHI. [W/m^2] Must be present in df.
        晴空 GHI 的列名。[瓦/平方米] 必须存在于 df 中。

    Returns
    -------
    k : np.ndarray
        Diffuse fraction k aligned to `df.index`. [unitless]
        与 `df.index` 对齐的散射分数 k。[无单位]

    Raises
    ------
    ValueError
        If ``ghi_clear_col`` is missing, ``period_minutes`` is unsupported, or
        ``df.index`` is not a :class:`~pandas.DatetimeIndex`.
        缺少 ``ghi_clear_col``、``period_minutes`` 不支持或 ``df.index`` 非 DatetimeIndex 时。
    """
    if ghi_clear_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain clear-sky column '{ghi_clear_col}'. "
            "Provide ghi_clear (e.g. :func:`bsrn.modeling.clear_sky.add_clearsky_columns`) before calling. / "
            f"数据框必须包含晴空列 '{ghi_clear_col}'。请先提供 ghi_clear（如通过 add_clearsky_columns）。"
        )

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
    cols = [ghi_col, ghi_clear_col]
    # Right-closed bins: (t-1h, t] so the value at 11:00 is the mean over (10:00, 11:00]
    resample_kw = {"rule": rule, "closed": "right", "label": "right"}
    counts = df[cols].resample(**resample_kw).count()
    bin_size = df.resample(**resample_kw).size()
    # Keep only those periods where more than half the data points are present for ghi_col
    enough_data = counts[ghi_col] > (bin_size / 2)
    # Compute mean over each right-closed period, set to NaN where not enough data
    df_rs = df[cols].resample(**resample_kw).mean()
    df_rs.loc[~enough_data] = np.nan
    df_rs = df_rs.dropna(subset=[ghi_col])

    # Use mid-interval timestamps for solar position so zenith/AST are representative of the hour,
    # avoiding systematic bias (e.g. end-of-hour sun making k trend monotonically through the day).
    period_td = pd.Timedelta(minutes=period_minutes)
    times_mid = df_rs.index - period_td / 2

    result = engerer2_separation(
        times_mid, df_rs[ghi_col].values, lat, lon,
        ghi_clear=df_rs[ghi_clear_col].values,
        averaging_period=period_minutes
    )
    # Align result back to hour-end index for assignment (result index is mid-interval).
    result = result.set_index(df_rs.index)

    # First-future assignment: for each 1-min t, use the hourly k at the end of the hour containing t
    # e.g. 1-min in (10:00, 11:00] get the k at 11:00 (mean over (10:00, 11:00])
    k_series = result["k"].reindex(df.index, method="bfill")
    return np.asarray(k_series, dtype=float)

def erbs_separation(times, ghi, lat, lon, elev=0, min_mu0=0.065,
                    max_zenith=87.0):
    """
    Erbs irradiance separation [1]_: diffuse fraction $k$ from clearness index $k_t$, then DHI and BNI.
    Erbs 辐照分离：由晴朗指数 $k_t$ 得散射分数 $k$，再得 DHI 与 BNI。

    Inputs are time, ghi, and location (lat, lon, elev); zenith and clearness index are computed inside.
    输入为时间、GHI 与位置（lat, lon, elev）；天顶角与晴朗指数在函数内计算。

    Piecewise formula (Erbs et al.):
    - $k_t \\leq 0.22$: $k = 1.0 - 0.09 k_t$
    - $0.22 < k_t \\leq 0.80$: $k = 0.9511 - 0.1604 k_t + 4.388 k_t^2 - 16.638 k_t^3 + 12.336 k_t^4$
    - $k_t > 0.80$: $k = 0.165$

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps. 
        时间戳。
    ghi : array-like
        Global horizontal irradiance. [W/m^2] 水平总辐照度。[瓦/平方米]
    lat : float
        Latitude. [degrees] 
        纬度。[度]
    lon : float
        Longitude. [degrees] 
        经度。[度]
    elev : float, default 0
        Site elevation. [m] Use same value as for zenith when comparing to pvlib.
        站点海拔。[米]。与 pvlib 比较时使用与 zenith 相同的值。
    min_mu0 : float, default 0.065
        Minimum $\\mu_0$ when computing $k_t$. 
        计算 $k_t$ 时 $\\mu_0$ 最小值。
    max_zenith : float, default 87.0
        Maximum zenith for valid BNI; beyond this BNI is set to 0. [degrees]
        BNI 有效的天顶角上限；超过则 BNI 置 0。[度]

    Returns
    -------
    out : pd.DataFrame
        DataFrame with index=times and columns ``k``, ``dhi``, ``bni`` (modeled).
        索引为 times、列为 k/dhi/bni（模型结果）的 DataFrame。

    Raises
    ------
    ValueError
        Propagated from :func:`_get_solar_and_kt` if ``times`` or ``ghi`` are invalid.
        由 :func:`_get_solar_and_kt` 在 ``times`` 或 ``ghi`` 无效时抛出。

    References
    ----------
    .. [1] Erbs, D. G., Klein, S. A., & Duffie, J. A. (1982). Estimation of
       the diffuse radiation fraction for hourly, daily and monthly-average
       global radiation. Solar Energy, 28(4), 293-302.
    """
    ghi, ghi_extra, zenith, mu0, kt, night = _get_solar_and_kt(
        times, ghi, lat, lon, elev=elev, min_mu0=min_mu0, max_clearness_index=1.0
    )

    # Calculate diffuse fraction / 计算散射分数 
    # For Kt <= 0.22, set the diffuse fraction / 对于 Kt <= 0.22，设置散射分数
    k = 1.0 - 0.09 * kt

    # For Kt > 0.22 and Kt <= 0.8, set the diffuse fraction
    # 对于 Kt > 0.22 且 Kt <= 0.8，设置散射分数
    k = np.where(
        (kt > 0.22) & (kt <= 0.8),
        0.9511 - 0.1604 * kt + 4.388 * kt ** 2
        - 16.638 * kt ** 3 + 12.336 * kt ** 4,
        k
    )

    # For Kt > 0.8, set the diffuse fraction / 对于 Kt > 0.8，设置散射分数
    k = np.where(kt > 0.8, 0.165, k)

    # This ensures the diffuse fraction is physically valid; values <0 or >1 are not physical.
    # 这确保了散射分数是物理有效的；值 <0 或 >1 不是物理有效的。
    k = np.clip(k, 0.0, 1.0)

    # Calculate DHI and BNI / 计算 DHI 与 BNI
    dhi, bni = _k_to_dhi_bni(ghi, k, zenith, max_zenith=max_zenith)

    # Nighttime separation handled by _k_to_dhi_bni and zenith limits.
    # 夜间处理由 _k_to_dhi_bni 和天顶角限制完成。
    return pd.DataFrame({"k": k, "dhi": dhi, "bni": bni}, index=times)

def brl_separation(times, ghi, lat, lon, min_mu0=0.065, max_zenith=87.0):
    """
    BRL irradiance separation [1]_: diffuse fraction $k$ from logistic function of
    $k_t$, AST, $\\alpha$, $K_t$, $\\psi$.
    BRL 辐照分离：由 $k_t$、AST、$\\alpha$、$K_t$、$\\psi$ 的逻辑回归函数得散射分数 $k$。

    $k = 1 / (1 + \\exp(-5.38 + 6.63 k_t + 0.006\\,\\text{AST} - 0.007\\,\\alpha
    + 1.75 K_t + 1.31 \\psi))$. $\\psi$ at sunrise = $k_{t+1}$, at sunset =
    $k_{t-1}$, else $(k_{t-1}+k_{t+1})/2$. $K_t$ = daily clearness index.
    日出时 $\\psi = k_{t+1}$，日落时 $\\psi = k_{t-1}$，否则为 $(k_{t-1}+k_{t+1})/2$。$K_t$ = 日晴朗指数。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps. 
        时间戳。
    ghi : array-like
        Global horizontal irradiance. [W/m^2] 
        水平总辐照度。[瓦/平方米]
    lat : float
        Latitude. [degrees] 
        纬度。[度]
    lon : float
        Longitude. [degrees] 
        经度。[度]
    min_mu0 : float, default 0.065
        Minimum $\\mu_0$ when computing $k_t$. 
        计算 $k_t$ 时 $\\mu_0$ 最小值。
    max_zenith : float, default 87.0
        Maximum zenith for valid BNI; beyond this BNI is set to 0. [degrees]
        BNI 有效的天顶角上限；超过则 BNI 置 0。[度]

    Returns
    -------
    out : pd.DataFrame
        DataFrame with index=times and columns ``k``, ``dhi``, ``bni`` (modeled).
        索引为 times、列为 k/dhi/bni（模型结果）的 DataFrame。

    Raises
    ------
    ValueError
        Propagated from :func:`_get_solar_and_kt` if ``times`` or ``ghi`` are invalid.
        由 :func:`_get_solar_and_kt` 在 ``times`` 或 ``ghi`` 无效时抛出。

    References
    ----------
    .. [1] Ridley, B., Boland, J., & Lauret, P. (2010). Modelling of
       diffuse solar fraction with multiple predictors. Renewable
       Energy, 35(2), 478-483.
    """
    ghi, ghi_extra, zenith, mu0, kt, night = _get_solar_and_kt(
        times, ghi, lat, lon, min_mu0=min_mu0, max_clearness_index=1.0
    )

    # Daily clearness index Kt (Eq. 7), only when enough daytime data / 日晴朗指数 Kt（仅在白天数据充足时）
    Kt = _brl_daily_clearness_index(times, ghi, ghi_extra, night)
    good_day = np.isfinite(Kt)

    # Apparent solar time AST (hours) / 地面太阳时 AST (小时)
    ast = _apparent_solar_time(times, lon)

    # Solar altitude alpha (degrees) = 90 - zenith / 太阳高度角 alpha (度) = 90 - 天顶角
    alpha = 90.0 - zenith

    # psi: piecewise from kt at adjacent timesteps / psi: 来自相邻时间步 kt 的分段
    dates = np.array([t.date() for t in times])
    psi = _brl_psi(kt, night, dates)
    # For days without a valid Kt, psi is not defined / 若该日无有效 Kt，则 psi 记为 NaN
    psi = np.where(good_day, psi, np.nan)

    # k = 1 / (1 + exp(...)); hourly kt and daily Kt / 逻辑回归公式
    exponent = (
        -5.38 + 6.63 * kt + 0.006 * ast - 0.007 * alpha
        + 1.75 * Kt + 1.31 * psi
    )
    with np.errstate(invalid="ignore", over="ignore"):
        k = 1.0 / (1.0 + np.exp(exponent))

    # This ensures the diffuse fraction is physically valid; values <0 or >1 are not physical.
    # 这确保了散射分数是物理有效的；值 <0 或 >1 不是物理有效的。
    k = np.clip(k, 0.0, 1.0)

    # Nighttime or days without valid Kt: k is NaN / 夜间或无有效 Kt 的日期：k 记为 NaN
    k = np.where(night | ~good_day, np.nan, k)

    # Calculate DHI and BNI / 计算 DHI 与 BNI
    dhi, bni = _k_to_dhi_bni(ghi, k, zenith, max_zenith=max_zenith, force_nan=~good_day)

    # Nighttime separation handled by _k_to_dhi_bni and zenith limits.
    # 夜间处理由 _k_to_dhi_bni 和天顶角限制完成。
    return pd.DataFrame({"k": k, "dhi": dhi, "bni": bni}, index=times)

def engerer2_separation(times, ghi, lat, lon, ghi_clear, averaging_period=1):
    """
    Engerer2 irradiance separation: estimate diffuse fraction ($k$), DHI and BNI from GHI.
    Engerer2 辐照分离：由 GHI 估算散射分数 ($k$)、DHI 与 BNI。

    Caller must provide clear-sky GHI (e.g. from a clear-sky model or add_clearsky_columns).
    调用方必须提供晴空 GHI（例如由晴空模型或 add_clearsky_columns 得到）。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps. 时间戳。
    ghi : array-like
        Global horizontal irradiance. [W/m^2] 水平总辐照度。[瓦/平方米]
    lat : float
        Latitude. [degrees] 纬度。[度]
    lon : float
        Longitude. [degrees] 经度。[度]
    ghi_clear : array-like
        Clear-sky GHI. [W/m^2] Same length as times. Required.
        晴空 GHI。[瓦/平方米] 与 times 等长。必填。
    averaging_period : int, default 1
        Coefficient set for resolution. [minutes] 1, 5, 10, 15, 30, 60, or 1440.
        对应分辨率的系数集。[分钟]

    Returns
    -------
    out : pd.DataFrame
        DataFrame with index=times and columns ``k``, ``dhi``, ``bni`` (modeled).
        索引为 times、列为 k/dhi/bni（模型结果）的 DataFrame。

    Raises
    ------
    ValueError
        If ``averaging_period`` is not in the supported set, ``times`` is not a
        :class:`~pandas.DatetimeIndex`, or ``ghi`` / ``ghi_clear`` lengths mismatch.
        ``averaging_period`` 不在支持集合、``times`` 非 DatetimeIndex 或 ``ghi``/``ghi_clear`` 长度不一致时。

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
    if not isinstance(times, pd.DatetimeIndex):
        raise ValueError("times must be a pd.DatetimeIndex.")
    ghi = np.asarray(ghi, dtype=float)
    if len(ghi) != len(times):
        raise ValueError("ghi must have the same length as times.")
    ghi_clear = np.asarray(ghi_clear, dtype=float)
    if len(ghi_clear) != len(times):
        raise ValueError("ghi_clear must have the same length as times.")

    ghi, ghi_extra, zenith, mu0, kt, night = _get_solar_and_kt(
        times, ghi, lat, lon, min_mu0=0.065, max_clearness_index=1.0
    )
    ast = _apparent_solar_time(times, lon)

    ghi_extra_safe = np.where(ghi_extra > 0, ghi_extra, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ktc = ghi_clear / ghi_extra_safe
    dktc = ktc - kt
    cloud_enh = np.where(ghi - ghi_clear > 0.015, ghi - ghi_clear, 0.0)
    # np.where evaluates both branches; divide(..., where=) skips ghi<=0 (no divide warning).
    k_de = np.zeros_like(ghi, dtype=float)
    np.divide(cloud_enh, ghi, out=k_de, where=ghi > 0)

    ktc = np.where(night, np.nan, ktc)
    dktc = np.where(night, np.nan, dktc)
    ast = np.where(night, np.nan, ast)
    k_de = np.where(night, np.nan, k_de)

    # Engerer2 logistic formula / Engerer2 逻辑公式
    c, b0, b1, b2, b3, b4, b5 = ENGERER2_PARAMS[averaging_period]
    with np.errstate(invalid="ignore"):
        k = c + (1 - c) / (1 + np.exp(
            b0 + b1 * kt + b2 * ast + b3 * zenith + b4 * dktc
        )) + b5 * k_de
    k = np.clip(k, 0.0, 1.0)

    dhi, bni = _k_to_dhi_bni(ghi, k, zenith, max_zenith=87.0)

    # Nighttime separation handled by _k_to_dhi_bni and zenith limits.
    # 夜间处理由 _k_to_dhi_bni 和天顶角限制完成。
    return pd.DataFrame({"k": k, "dhi": dhi, "bni": bni}, index=times)

def yang4_separation(times, ghi, lat, lon, ghi_clear):
    """
    Yang4 irradiance separation: diffuse fraction k_d from k_t, AST, Z, Δk_tc, k_de, and Engerer2 60-min k.
    k_d^YANG4 = C + (1-C)/(1 + exp(β0 + β1*k_t + β2*AST + β3*Z + β4*Δk_tc + β6*k_d,60min^ENGERER2)) + β5*k_de.
    Uses YANG2 coefficient set (TABLE III) from 1-min SURFRAD data.

    Caller must provide clear-sky GHI (e.g. from a clear-sky model or add_clearsky_columns).
    调用方必须提供晴空 GHI（例如由晴空模型或 add_clearsky_columns 得到）。

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps. 时间戳。
    ghi : array-like
        Global horizontal irradiance. [W/m^2] 水平总辐照度。[瓦/平方米]
    lat : float
        Latitude. [degrees] 纬度。[度]
    lon : float
        Longitude. [degrees] 经度。[度]
    ghi_clear : array-like
        Clear-sky GHI. [W/m^2] Same length as times. Required.
        晴空 GHI。[瓦/平方米] 与 times 等长。必填。

    Returns
    -------
    out : pd.DataFrame
        DataFrame with index=times and columns ``k``, ``dhi``, ``bni`` (modeled).
        索引为 times、列为 k/dhi/bni（模型结果）的 DataFrame。

    Raises
    ------
    ValueError
        If ``times`` is not a :class:`~pandas.DatetimeIndex`, lengths mismatch, or
        :func:`_engerer2_k_at_resolution` rejects inputs (e.g. missing ``ghi_clear``).
        ``times`` 非 DatetimeIndex、长度不一致或 :func:`_engerer2_k_at_resolution` 拒绝输入（如缺 ``ghi_clear``）时。

    References
    ----------
    .. [1] Yang, D. (2021). Temporal-resolution cascade model for
       separation of 1-min beam and diffuse irradiance. Journal of
       Renewable and Sustainable Energy, 13(5), 053703.
    .. [2] Yang, D., & Boland, J. (2019). Satellite-augmented diffuse
       solar radiation separation models. Journal of Renewable and
       Sustainable Energy, 11(2), 023704.
    """
    if not isinstance(times, pd.DatetimeIndex):
        raise ValueError("times must be a pd.DatetimeIndex.")
    ghi = np.asarray(ghi, dtype=float)
    if len(ghi) != len(times):
        raise ValueError("ghi must have the same length as times.")
    ghi_clear = np.asarray(ghi_clear, dtype=float)
    if len(ghi_clear) != len(times):
        raise ValueError("ghi_clear must have the same length as times.")

    # Build minimal df for _engerer2_k_at_resolution (resampling)
    df_min = pd.DataFrame({"ghi": ghi, "ghi_clear": ghi_clear}, index=times)
    k_engerer2_60 = _engerer2_k_at_resolution(
        df_min, lat, lon, 60, ghi_col="ghi", ghi_clear_col="ghi_clear"
    )

    ghi, ghi_extra, zenith, mu0, kt, night = _get_solar_and_kt(
        times, ghi, lat, lon, min_mu0=0.065, max_clearness_index=1.0
    )
    ast = _apparent_solar_time(times, lon)

    ghi_extra_safe = np.where(ghi_extra > 0, ghi_extra, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi / ghi_extra_safe
        ktc = ghi_clear / ghi_extra_safe
    dktc = ktc - kt
    cloud_enh = np.where(ghi - ghi_clear > 0.015, ghi - ghi_clear, 0.0)
    k_de = np.zeros_like(ghi, dtype=float)
    np.divide(cloud_enh, ghi, out=k_de, where=ghi > 0)

    dktc = np.where(night, np.nan, dktc)
    ast = np.where(night, np.nan, ast)
    k_de = np.where(night, np.nan, k_de)
    k_engerer2_60 = np.where(night, np.nan, k_engerer2_60)

    # Yang4 logistic formula / Yang4 逻辑公式
    C, b0, b1, b2, b3, b4, b5, b6 = YANG4_PARAMS
    exponent = b0 + b1 * kt + b2 * ast + b3 * zenith + b4 * dktc + b6 * k_engerer2_60
    with np.errstate(invalid="ignore", over="ignore"):
        k = C + (1 - C) / (1 + np.exp(exponent)) + b5 * k_de
    k = np.clip(k, 0.0, 1.0)

    dhi, bni = _k_to_dhi_bni(ghi, k, zenith, max_zenith=87.0)

    # Nighttime separation handled by _k_to_dhi_bni and zenith limits.
    # 夜间处理由 _k_to_dhi_bni 和天顶角限制完成。
    return pd.DataFrame({"k": k, "dhi": dhi, "bni": bni}, index=times)
