"""
Clear-sky detection (CSD) utilities.
晴空检测（CSD）工具函数。

Implements selected CSD methods from the MATLAB csd-library with a common
Python output interface.
实现 MATLAB csd-library 中的部分 CSD 方法，并提供统一的 Python 输出接口。

Original MATLAB implementations and methodology definitions are from
Jamie M. Bright's clear-sky detection library:
https://github.com/JamieMBright/csd-library
原始 MATLAB 实现和方法定义来自 Jamie M. Bright 的晴空检测库：
https://github.com/JamieMBright/csd-library
"""

import warnings
import numpy as np
import pandas as pd


def _as_1d_array(x, name, n=None):
    """
    Convert input to 1D float array and validate length.
    将输入转为一维浮点数组并校验长度。

    Parameters
    ----------
    x : array-like
        Input to convert.
        待转换的输入。
    name : str
        Name for error messages.
        错误信息中的名称。
    n : int, optional
        Required length; if set, raise when len(x) != n.
        要求长度；若设置则校验。

    Returns
    -------
    np.ndarray
        1D float array.
        一维浮点数组。

    Raises
    ------
    ValueError
        When n is set and len(x) != n.
        当 n 已设置且 len(x) != n 时。
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and len(arr) != n:
        raise ValueError(f"{name} must have length {n}. / {name} 长度必须为 {n}。")
    return arr


def _resolve_index(times, n):
    """
    Resolve output index.
    解析输出索引。

    Parameters
    ----------
    times : array-like, pd.DatetimeIndex, or None
        Time index; if None, return integer range.
        时间索引；None 则返回整数范围。
    n : int
        Expected length.
        期望长度。

    Returns
    -------
    pd.RangeIndex or pd.DatetimeIndex
        Index of length n.
        长度为 n 的索引。

    Raises
    ------
    ValueError
        When len(times) != n.
        当 len(times) != n 时。
    """
    if times is None:
        return pd.RangeIndex(start=0, stop=n, step=1, name="time")
    if isinstance(times, pd.DatetimeIndex):
        if len(times) != n:
            raise ValueError("times length must match inputs. / times 长度必须与输入一致。")
        return times
    idx = pd.DatetimeIndex(times)
    if len(idx) != n:
        raise ValueError("times length must match inputs. / times 长度必须与输入一致。")
    return idx


def _hankel_window(values, window):
    """
    Build MATLAB-style Hankel windows with trailing NaN padding.
    构造 MATLAB 风格 Hankel 窗口并在尾部填充 NaN。

    Equivalent to MATLAB: to avoid a loop, the time series are concatenated
    with a tail so one matrix call builds all windows, e.g.
    ``ghi_window = hankel(ghi, [ghi(end), NaN(1, interval_size-1)])``.
    Each row i is the window [values[i], values[i+1], ..., values[i+window-1]],
    with NaN padding where the window extends past the end of the series.
    等价于 MATLAB：为避免循环，将时间序列与尾部拼接后一次构造所有窗口。

    Parameters
    ----------
    values : array-like
        1D time series.
        一维时间序列。
    window : int
        Window length (number of columns).
        窗口长度（列数）。

    Returns
    -------
    np.ndarray
        Shape (len(values), window); row i = window starting at index i.
        形状 (len(values), window)。
    """
    n = len(values)
    mat = np.full((n, window), np.nan, dtype=float)
    for i in range(window):
        end = n - i
        if end > 0:
            mat[:end, i] = values[i:]
    return mat


def _window_sufficient_valid(window_mat):
    """
    True where each row has more than half non-NaN (e.g. window=10 -> at least 5).
    每行非 NaN 超过半数时为 True（如 window=10 则至少 5 个）。

    Parameters
    ----------
    window_mat : np.ndarray
        2D array, shape (n, window).
        二维数组，形状 (n, window)。

    Returns
    -------
    np.ndarray of bool
        True where row has >= (window+1)//2 finite values.
        每行有限值数 >= (window+1)//2 为 True。
    """
    n_valid = np.sum(np.isfinite(window_mat), axis=1)
    min_required = (window_mat.shape[1] + 1) // 2
    return n_valid >= min_required



def _safe_divide(num, den):
    """
    Safe division with NaN on invalid entries.
    安全除法，对无效位置返回 NaN。

    Parameters
    ----------
    num : array-like
        Numerator.
        分子。
    den : array-like
        Denominator.
        分母。

    Returns
    -------
    np.ndarray
        num/den with non-finite results set to NaN.
        商，非有限处为 NaN。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def _nanmax_no_warn(arr, axis):
    """
    nanmax without emitting all-NaN runtime warnings.
    计算 nanmax 且不发出全 NaN 运行时警告。

    Parameters
    ----------
    arr : np.ndarray
        Input array.
        输入数组。
    axis : int
        Axis over which to take maximum.
        沿该轴取最大值。

    Returns
    -------
    np.ndarray
        np.nanmax(arr, axis=axis).
        沿轴 nanmax 结果。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmax(arr, axis=axis)
    return out


def _nanstd_no_warn(arr, axis, ddof=0):
    """
    nanstd without emitting empty-slice runtime warnings.
    计算 nanstd 且不发出空切片运行时警告。

    Parameters
    ----------
    arr : np.ndarray
        Input array.
        输入数组。
    axis : int
        Axis over which to take standard deviation.
        沿该轴计算标准差。
    ddof : int, default 0
        Delta degrees of freedom for std.
        标准差自由度。

    Returns
    -------
    np.ndarray
        np.nanstd(arr, axis=axis, ddof=ddof).
        沿轴 nanstd 结果。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanstd(arr, axis=axis, ddof=ddof)
    return out


def _csd_to_output(index, cloud_flag, method, diagnostics=None,
                   return_diagnostics=False):
    """
    Standardize CSD outputs.
    标准化 CSD 输出。

    Parameters
    ----------
    index : pd.Index
        Output index.
        输出索引。
    cloud_flag : array-like
        0=clear, 1=cloudy, NaN=invalid.
        0 晴空，1 有云，NaN 无效。
    method : str
        Method name for output column.
        方法名（输出列）。
    diagnostics : dict, optional
        Extra arrays to add to output.
        附加诊断数组。
    return_diagnostics : bool, default False
        If True, include diagnostics in output.
        为 True 时输出含诊断列。

    Returns
    -------
    pd.DataFrame
        Columns: is_clearsky, cloud_flag, method [, diagnostics].
        列：is_clearsky, cloud_flag, method 等。
    """
    cloud = np.asarray(cloud_flag, dtype=float)
    is_clearsky = pd.array(cloud == 0, dtype="boolean")
    is_clearsky[np.isnan(cloud)] = pd.NA

    out = pd.DataFrame(
        {
            "is_clearsky": is_clearsky,
            "cloud_flag": cloud,
            "method": method,
        },
        index=index,
    )
    if return_diagnostics and diagnostics is not None:
        for col, arr in diagnostics.items():
            out[col] = np.asarray(arr, dtype=float)
    return out


def _brightsun_component_flag(meas, clear, zenith, window=10, is_ghi=True,
                              return_diagnostics=False):
    """
    Bright-Sun component-level clear-sky test for one irradiance component.
    Bright-Sun 分量级晴空判定，用于单一辐照分量。

    This helper implements the per-component criteria used inside the
    Bright-Sun tri-component method [1]_ (for either GHI or DHI):
    本函数实现 Bright-Sun 三分量方法 [1] 中对单个分量（GHI 或 DHI）的判据：

    - Sliding-window statistics (length ``window`` minutes) on measured and
      clear-sky series.
      在实测和晴空时间序列上计算长度为 ``window`` 分钟的滑动窗口统计量。
    - Relative mean and max differences (|meas-clear|/clear) with
      zenith-dependent thresholds (:math:`c1`, :math:`c2`).
      利用相对均值和相对最大值差（|实测-晴空|/晴空），并随天顶角调整阈值
      （:math:`c1`, :math:`c2`）。
    - Normalized line-length difference of the time series (meas vs clear)
      with a zenith-dependent acceptance band (:math:`c3`).
      使用归一化折线长度差（实测与晴空）并施加随天顶角变化的接受区间
      （:math:`c3`）。
    - Normalized slope variability (windowed std of first differences
      divided by mean) with threshold :math:`\\sigma_{\\mathrm{lim}}` (:math:`c4`).
      归一化斜率变异度（差分标准差/均值）并与阈值 :math:`\\sigma_{\\mathrm{lim}}`
      （:math:`c4`）比较。
    - Maximum absolute point-wise slope difference (X) with zenith-dependent
      limit (:math:`c5`).
      点对点斜率差的最大绝对值 X，并施加随天顶角变化的上限
      （:math:`c5`）。
    - Valid-clear check on clear-sky mean (:math:`c6`).
      对晴空均值进行有效性检查（:math:`c6`），剔除无效或为零的晴空估计。

    For each time step, **1 = cloud, 0 = clear** at the component level.
    When all six criteria accept the point, it is considered clear for that
    component.
    在分量级别上，每个时间步 **1 表示有云，0 表示晴空**。
    当六个判据全部通过时，该时间步在该分量上视为晴空。

    Parameters
    ----------
    meas : array-like
        Measured component (GHI or DHI), 1D.
        实测分量（GHI 或 DHI），一维数组。
    clear : array-like
        Corresponding clear-sky estimate for the same component.
        对应的晴空估计时间序列。
    zenith : array-like
        Solar zenith angle [degrees].
        太阳天顶角 [度]。
    window : int, default 10
        Sliding-window length (minutes) used to build Hankel matrices.
        构建 Hankel 矩阵的滑动窗口长度（分钟）。
    is_ghi : bool, default True
        If True, apply Bright-Sun GHI parameterization; otherwise use the
        DHI parameterization.
        若为 True，使用 GHI 的 Bright-Sun 参数化；否则使用 DHI 参数化。
    return_diagnostics : bool, default False
        If True, also return a diagnostics dict with individual criteria,
        thresholds, and intermediates.
        若为 True，返回包含各判据、阈值和中间量的诊断字典。

    Returns
    -------
    cloud_flag : np.ndarray
        1 = cloudy, 0 = clear, NaN = invalid input.
        1 表示有云，0 表示晴空，NaN 表示输入无效。
    diagnostics : dict, optional
        Only when ``return_diagnostics=True``; contains arrays for c1–c6,
        intermediate statistics, and zenith-dependent thresholds.
        仅当 ``return_diagnostics=True`` 时返回；包含 c1–c6 判据、相关统计量
        以及随天顶角变化的阈值数组。

    References
    ----------
    .. [1] Bright, J. M., Sun, X., Gueymard, C. A., Acord, B., Wang, P., &
       Engerer, N. A. (2020). Bright-Sun: A globally applicable 1-min
       irradiance clear-sky detection model. Renewable and Sustainable
       Energy Reviews, 121, 109706.
    """
    meas = _as_1d_array(meas, "meas")
    clear = _as_1d_array(clear, "clear", n=len(meas))
    zenith = _as_1d_array(zenith, "zenith", n=len(meas))
    n = len(meas)

    # Zenith grid 20–90°, 0.01 step; turnaround at 30° (MATLAB zenith_turn_around)
    # 天顶角网格 20–90°，步长 0.01°；30° 为参数转折点（与 MATLAB 中 zenith_turn_around 一致）。
    z_ref = np.arange(20.0, 90.01, 0.01)
    n1 = int(round((30.0 - 20.0) / 0.01)) + 1
    if is_ghi:
        c1_lim_arr = np.flip(np.concatenate([
            np.linspace(0.5, 0.125, n1),
            np.linspace(0.125, 0.25, len(z_ref) - n1),
        ]))
        c2_lim_arr = np.flip(np.concatenate([
            np.linspace(0.5, 0.125, n1),
            np.linspace(0.125, 0.25, len(z_ref) - n1),
        ]))
        c3_lower_arr = np.flip(np.concatenate([
            np.linspace(-0.5, -0.5, n1),
            np.linspace(-0.5, -7.0, len(z_ref) - n1),
        ]))
        sigma_lim = 0.4
        c5_slope = 15.0
        c5_small = 45.0
    else:
        c1_lim_arr = np.flip(np.concatenate([
            np.linspace(0.5, 0.5, n1),
            np.linspace(0.5, 0.25, len(z_ref) - n1),
        ]))
        c2_lim_arr = np.flip(np.concatenate([
            np.linspace(0.5, 0.5, n1),
            np.linspace(0.5, 0.25, len(z_ref) - n1),
        ]))
        c3_lower_arr = np.flip(np.concatenate([
            np.linspace(-1.7, -1.7, n1),
            np.linspace(-1.7, -6.0, len(z_ref) - n1),
        ]))
        sigma_lim = 0.2
        c5_slope = 8.0
        c5_small = 24.0

    c3_upper_arr = np.abs(c3_lower_arr)
    c5_lim_arr = np.flip(np.concatenate([
        np.linspace(c5_slope, c5_slope, n1),
        np.linspace(c5_slope, c5_small, len(z_ref) - n1),
    ]))

    inds = np.searchsorted(z_ref, np.clip(zenith, z_ref[0], z_ref[-1]), side="left")
    inds = np.clip(inds, 0, len(z_ref) - 1)
    c1_lim = c1_lim_arr[inds]
    c2_lim = c2_lim_arr[inds]
    c3_lower = c3_lower_arr[inds]
    c3_upper = c3_upper_arr[inds]
    c5_lim = c5_lim_arr[inds]

    # Build Hankel matrices: each row = length-`window` time series segment,
    # used to compute local statistics over sliding windows.
    # 构造 Hankel 矩阵：每行对应一个长度为 window 的时间段，用于局部统计量计算。
    m_win = _hankel_window(meas, window)
    c_win = _hankel_window(clear, window)
    sufficient_m = _window_sufficient_valid(m_win)
    sufficient_c = _window_sufficient_valid(c_win)

    # Window-mean and window-max for measured / clear-sky component.
    # 窗口内均值和最大值（实测与晴空）。
    meas_mean = np.nanmean(m_win, axis=1)
    clear_mean = np.nanmean(c_win, axis=1)
    meas_max = _nanmax_no_warn(m_win, axis=1)
    clear_max = _nanmax_no_warn(c_win, axis=1)
    meas_mean[~sufficient_m] = np.nan
    clear_mean[~sufficient_c] = np.nan
    meas_max[~sufficient_m] = np.nan
    clear_max[~sufficient_c] = np.nan

    # First differences for slope-based criteria and line-length.
    # 一阶差分（斜率相关判据与线长度使用）。
    # MATLAB: diff along axis-0 (rows) + NaN-row pad → (n, window)
    # MATLAB：沿 axis-0（行）差分 + NaN 行填充 → (n, window)
    _pad = lambda m: np.vstack([np.diff(m, axis=0),
                                np.full((1, m.shape[1]), np.nan)])
    meas_diff = _pad(m_win)
    clear_diff = _pad(c_win)
    meas_slope_nstd = _safe_divide(
        _nanstd_no_warn(meas_diff, axis=1, ddof=0), meas_mean)
    meas_line = np.nansum(
        np.sqrt(meas_diff * meas_diff + 1.0), axis=1)
    clear_line = np.nansum(
        np.sqrt(clear_diff * clear_diff + 1.0), axis=1)
    meas_slope_nstd[~sufficient_m] = np.nan
    meas_line[~sufficient_m] = np.nan
    clear_line[~sufficient_c] = np.nan

    line_diff_norm = _safe_divide(
        meas_line - clear_line, clear_line)
    X = _nanmax_no_warn(
        np.abs(meas_diff - clear_diff), axis=1)
    line_diff_norm[~sufficient_m | ~sufficient_c] = np.nan
    X[~sufficient_m | ~sufficient_c] = np.nan

    # 1=cloud by default; 0=clear when criterion passes (MATLAB: c1=0 when rel diff < lim)
    # 默认认为有云（1），当各自条件满足时置为 0（晴空）。
    c1 = np.ones(n, dtype=float)
    c1[_safe_divide(np.abs(meas_mean - clear_mean), clear_mean) < c1_lim] = 0.0
    c2 = np.ones(n, dtype=float)
    c2[_safe_divide(np.abs(meas_max - clear_max), clear_max) < c2_lim] = 0.0
    c3 = np.ones(n, dtype=float)
    c3[(line_diff_norm > c3_lower) & (line_diff_norm < c3_upper)] = 0.0
    c4 = np.ones(n, dtype=float)
    c4[meas_slope_nstd < sigma_lim] = 0.0
    c5 = np.ones(n, dtype=float)
    c5[X < c5_lim] = 0.0
    c6 = np.ones(n, dtype=float)
    c6[(clear_mean != 0) & np.isfinite(clear_mean)] = 0.0

    cloud_flag = ((c1 + c2 + c3 + c4 + c5 + c6) > 0).astype(float)
    bad = np.isnan(meas) | np.isnan(clear)
    cloud_flag[bad] = np.nan
    if not return_diagnostics:
        return cloud_flag
    diagnostics = {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "c5": c5,
        "c6": c6,
        "meas_mean": meas_mean,
        "clear_mean": clear_mean,
        "meas_max": meas_max,
        "clear_max": clear_max,
        "line_diff_norm": line_diff_norm,
        "meas_slope_nstd": meas_slope_nstd,
        "X": X,
        "c1_lim": c1_lim,
        "c2_lim": c2_lim,
        "c3_lower": c3_lower,
        "c3_upper": c3_upper,
        "c5_lim": c5_lim,
    }
    for key, arr in diagnostics.items():
        a = np.asarray(arr, dtype=float)
        a[bad] = np.nan
        diagnostics[key] = a
    return cloud_flag, diagnostics


def _reno_cloud_flag(ghi, ghi_clear, window=10, mean_lim=75.0, max_lim=75.0,
                     lower_L_lim=-5.0, upper_L_lim=10.0, sigma_lim=0.005,
                     X_lim=8.0):
    """
    Core Reno-style five-criteria cloud flag with an additional valid-clear
    criterion, following csd-library Reno2016CSD logic.
    Reno 风格五准则云标记（含有效晴空附加准则），遵循 csd-library Reno2016CSD 逻辑。

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance. [W/m^2]
        水平总辐照度。[瓦/平方米]
    ghi_clear : array-like
        Clear-sky GHI. [W/m^2]
        晴空水平总辐照度。[瓦/平方米]
    window : int, default 10
        Window size for criteria.
        判据窗口长度。
    mean_lim, max_lim, lower_L_lim, upper_L_lim, sigma_lim, X_lim : float
        Thresholds for the five criteria and line-diff bounds.
        五准则及线差阈值。

    Returns
    -------
    cloud_flag : np.ndarray
        0=clear, 1=cloudy, NaN=invalid.
        0 晴空，1 有云，NaN 无效。
    diagnostics : dict
        Arrays for each criterion and intermediates.
        各判据及中间量。
    """
    ghi = _as_1d_array(ghi, "ghi")
    ghi_clear = _as_1d_array(ghi_clear, "ghi_clear", n=len(ghi))
    n = len(ghi)

    # Build sliding-window matrices (each row = one window of length `window`).
    # 构造滑动窗口矩阵（每行为一个长度为 window 的窗口）。
    ghi_window = _hankel_window(ghi, window)
    clear_window = _hankel_window(ghi_clear, window)

    # Require more than half of window points to be valid; else treat stats as invalid.
    # 要求窗口内超过半数点为有效，否则该窗口的统计量视为无效。
    sufficient_ghi = _window_sufficient_valid(ghi_window)
    sufficient_clear = _window_sufficient_valid(clear_window)

    # Window mean and max: used in criteria c1 (mean difference) and c2 (max difference).
    # 窗口均值与最大值：用于判据 c1（均值差）与 c2（最大值差）。
    meas_mean = np.nanmean(ghi_window, axis=1)
    clear_mean = np.nanmean(clear_window, axis=1)
    meas_max = _nanmax_no_warn(ghi_window, axis=1)
    clear_max = _nanmax_no_warn(clear_window, axis=1)
    meas_mean[~sufficient_ghi] = np.nan
    clear_mean[~sufficient_clear] = np.nan
    meas_max[~sufficient_ghi] = np.nan
    clear_max[~sufficient_clear] = np.nan

    # Along-window differences and derived stats: slope variability (c4) and line length (c3).
    # 沿窗口的差分及派生量：斜率变异性（c4）与折线长度差（c3）。
    # MATLAB: diff along axis-0 (rows) + NaN-row pad → (n, window)
    # MATLAB：沿 axis-0（行）差分 + NaN 行填充 → (n, window)
    _pad = lambda m: np.vstack([np.diff(m, axis=0),
                                np.full((1, m.shape[1]), np.nan)])
    meas_diff = _pad(ghi_window)
    clear_diff = _pad(clear_window)

    meas_slope_nstd = _safe_divide(
        _nanstd_no_warn(meas_diff, axis=1, ddof=0), meas_mean)
    meas_line = np.nansum(
        np.sqrt(meas_diff * meas_diff + 1.0), axis=1)
    clear_line = np.nansum(
        np.sqrt(clear_diff * clear_diff + 1.0), axis=1)
    meas_slope_nstd[~sufficient_ghi] = np.nan
    meas_line[~sufficient_ghi] = np.nan
    clear_line[~sufficient_clear] = np.nan

    line_diff = meas_line - clear_line
    X = _nanmax_no_warn(
        np.abs(meas_diff - clear_diff), axis=1)
    line_diff[~sufficient_ghi | ~sufficient_clear] = np.nan
    X[~sufficient_ghi | ~sufficient_clear] = np.nan

    # Five Reno criteria plus valid-clear check: any True -> cloudy (1).
    # 五个 Reno 判据加有效晴空检查：任一为 True 则判为有云 (1)。
    c1 = np.abs(meas_mean - clear_mean) > mean_lim      # mean GHI vs clear-sky
    c2 = np.abs(meas_max - clear_max) > max_lim          # max GHI vs clear-sky
    c3 = (line_diff < lower_L_lim) | (line_diff > upper_L_lim)  # line-length difference
    c4 = meas_slope_nstd > sigma_lim                    # normalized slope std
    c5 = X > X_lim                                      # max absolute diff
    c6 = (clear_mean == 0) | (~np.isfinite(clear_mean))  # invalid or zero clear-sky

    cloud_flag = (c1 | c2 | c3 | c4 | c5 | c6).astype(float)
    bad = np.isnan(ghi) | np.isnan(ghi_clear)
    cloud_flag[bad] = np.nan

    # Package diagnostics and mask invalid input rows.
    # 打包诊断量并屏蔽无效输入行。
    diagnostics = {
        "c1": c1.astype(float),
        "c2": c2.astype(float),
        "c3": c3.astype(float),
        "c4": c4.astype(float),
        "c5": c5.astype(float),
        "c6": c6.astype(float),
        "meas_mean": meas_mean,
        "clear_mean": clear_mean,
        "meas_max": meas_max,
        "clear_max": clear_max,
        "line_diff": line_diff,
        "meas_slope_nstd": meas_slope_nstd,
        "X": X,
    }
    for key in diagnostics:
        arr = np.asarray(diagnostics[key], dtype=float)
        arr[bad] = np.nan
        diagnostics[key] = arr
    return cloud_flag, diagnostics


def reno_csd(ghi, ghi_clear, times=None, return_diagnostics=False):
    """
    Reno2016 clear-sky detection.
    Reno2016 晴空检测 [1]_。

    MATLAB mapping: `Reno2016CSD(ghi, ghics, plot_figure)`.
    MATLAB 变量映射：`ghics -> ghi_clear`。

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance (`ghi`). [W/m^2]
        水平总辐照度 (`ghi`)。[瓦/平方米]
    ghi_clear : array-like
        Clear-sky global horizontal irradiance (`ghics -> ghi_clear`). [W/m^2]
        晴空水平总辐照度 (`ghics -> ghi_clear`)。[瓦/平方米]
    times : array-like or pd.DatetimeIndex, optional
        Time index for outputs.
        输出时间索引。
    return_diagnostics : bool, default False
        If True, include method diagnostics.
        若为 True，返回方法诊断量。

    Returns
    -------
    out : pd.DataFrame
        Standardized output with `is_clearsky`, `cloud_flag`, and optional diagnostics.
        标准化输出，含 `is_clearsky`、`cloud_flag` 及可选诊断量。

    Raises
    ------
    ValueError
        When input lengths do not match.
        输入长度不一致时。

    References
    ----------
    .. [1] Reno, M. J., & Hansen, C. W. (2016). Identification of periods of clear sky
       irradiance in time series of GHI measurements. Renewable Energy, 90, 520-531.
    """
    ghi = _as_1d_array(ghi, "ghi")
    idx = _resolve_index(times, len(ghi))
    cloud_flag, diagnostics = _reno_cloud_flag(ghi, ghi_clear)
    return _csd_to_output(idx, cloud_flag, "reno", diagnostics, return_diagnostics)


def ineichen_csd(ghi, ghi_extra, zenith, times=None, return_diagnostics=False):
    """
    Ineichen2009 clear-sky detection.
    Ineichen2009 晴空检测 [1]_。

    MATLAB mapping: `Ineichen2009CSD(ghi, exth, zen, plot_figure)`.
    MATLAB 变量映射：`exth -> ghi_extra`, `zen -> zenith`。
    Convention: MATLAB csd(kt_prime<0.65)=1 marks low clearness (cloudy); here cloud_flag 0=clear,
    1=cloudy, so cloud_flag=1 where kt_prime<0.65.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance (`ghi`). [W/m^2]
        水平总辐照度 (`ghi`)。[瓦/平方米]
    ghi_extra : array-like
        Horizontal extraterrestrial irradiance (`exth -> ghi_extra`). [W/m^2]
        地外水平辐照度 (`exth -> ghi_extra`)。[瓦/平方米]
    zenith : array-like
        Solar zenith angle (`zen -> zenith`). [degrees]
        太阳天顶角 (`zen -> zenith`)。[度]
    times : array-like or pd.DatetimeIndex, optional
        Time index for outputs.
        输出时间索引。
    return_diagnostics : bool, default False
        If True, include method diagnostics.
        若为 True，返回方法诊断量。

    Returns
    -------
    out : pd.DataFrame
        Standardized output with `is_clearsky`, `cloud_flag`, and optional diagnostics.
        标准化输出，含 `is_clearsky`、`cloud_flag` 及可选诊断量。

    Raises
    ------
    ValueError
        When input lengths do not match.
        输入长度不一致时。

    References
    ----------
    .. [1] Ineichen, P., Barroso, C. S., Geiger, B., Hollmann, R., Marsouin, A., &
       Mueller, R. (2009). Satellite Application Facilities irradiance products:
       hourly time step comparison and validation over Europe. International
       Journal of Remote Sensing, 30(21), 5549-5571.
    """
    ghi = _as_1d_array(ghi, "ghi")
    ghi_extra = _as_1d_array(ghi_extra, "ghi_extra", n=len(ghi))
    zenith = _as_1d_array(zenith, "zenith", n=len(ghi))
    idx = _resolve_index(times, len(ghi))

    kt = _safe_divide(ghi, ghi_extra)
    h = 90.0 - zenith
    with np.errstate(divide="ignore", invalid="ignore"):
        M = 1.0 / (np.sin(np.radians(h)) + 0.15 * np.power(h + 3.885, -1.253))
        kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / M)) + 0.1)

    # Convention: Python cloud_flag 0=clear, 1=cloudy. In the reference MATLAB, csd(kt_prime<0.65)=1
    # assigns 1 where modified clearness is low (cloudy). So: kt_prime < 0.65 -> cloudy (cloud_flag=1).
    cloud_flag = np.zeros(len(ghi), dtype=float)
    cloud_flag[kt_prime < 0.65] = 1.0
    bad = (np.isnan(ghi) | np.isnan(ghi_extra)
           | np.isnan(zenith) | (zenith >= 90.0)
           | (ghi_extra <= 0.0))
    cloud_flag[bad] = np.nan

    diagnostics = {"kt": kt, "M": M, "kt_prime": kt_prime}
    for key in diagnostics:
        arr = np.asarray(diagnostics[key], dtype=float)
        arr[bad] = np.nan
        diagnostics[key] = arr
    return _csd_to_output(idx, cloud_flag, "ineichen", diagnostics, return_diagnostics)


def lefevre_csd(ghi, dhi, ghi_extra, zenith, times=None,
                return_diagnostics=False):
    """
    Lefevre2013 clear-sky detection.
    Lefevre2013 晴空检测 [1]_。

    MATLAB mapping: `Lefevre2013CSD(ghi, dif, exth, zen, plot_figure)`.
    MATLAB 变量映射：`dif -> dhi`, `exth -> ghi_extra`, `zen -> zenith`。

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance (`ghi`). [W/m^2]
        水平总辐照度 (`ghi`)。[瓦/平方米]
    dhi : array-like
        Diffuse horizontal irradiance (`dif -> dhi`). [W/m^2]
        水平散射辐照度 (`dif -> dhi`)。[瓦/平方米]
    ghi_extra : array-like
        Horizontal extraterrestrial irradiance (`exth -> ghi_extra`). [W/m^2]
        地外水平辐照度 (`exth -> ghi_extra`)。[瓦/平方米]
    zenith : array-like
        Solar zenith angle (`zen -> zenith`). [degrees]
        太阳天顶角 (`zen -> zenith`)。[度]
    times : array-like or pd.DatetimeIndex, optional
        Time index for outputs.
        输出时间索引。
    return_diagnostics : bool, default False
        If True, include method diagnostics.
        若为 True，返回方法诊断量。

    Returns
    -------
    out : pd.DataFrame
        Standardized output with `is_clearsky`, `cloud_flag`, and optional diagnostics.
        标准化输出，含 `is_clearsky`、`cloud_flag` 及可选诊断量。

    Raises
    ------
    ValueError
        When input lengths do not match.
        输入长度不一致时。

    References
    ----------
    .. [1] Lefèvre, M., Oumbe, A., Blanc, P., Espinar, B., Gschwind, B., Qu, Z., ...
       & Morcrette, J. J. (2013). McClear: a new model estimating downwelling
       solar radiation at ground level in clear-sky conditions. Atmospheric
       Measurement Techniques, 6(9), 2403-2418.
    """
    ghi = _as_1d_array(ghi, "ghi")
    dhi = _as_1d_array(dhi, "dhi", n=len(ghi))
    ghi_extra = _as_1d_array(ghi_extra, "ghi_extra", n=len(ghi))
    zenith = _as_1d_array(zenith, "zenith", n=len(ghi))
    idx = _resolve_index(times, len(ghi))

    cloud_flag = np.ones(len(ghi), dtype=float)
    k = _safe_divide(dhi, ghi)
    # Invalidate k where GHI or DHI are non-physical (negative or zero GHI).
    # 当 GHI 或 DHI 非物理（负值或 GHI 为零）时，将 k 视为无效。
    non_physical = (ghi <= 0) | (dhi < 0)
    k[non_physical] = np.nan
    first_filter_clear = k < 0.3
    cloud_flag[first_filter_clear] = 0.0

    kt = _safe_divide(ghi, ghi_extra)
    h = 90.0 - zenith
    with np.errstate(divide="ignore", invalid="ignore"):
        M = 1.0 / (np.sin(np.radians(h)) + 0.15 * np.power(h + 3.885, -1.253))
        kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / M)) + 0.1)

    # Lefevre second filter: require enough first-filter-retained points
    # in both [t-90, t] and [t, t+90], then evaluate std of kt' over [t-90, t+90].
    retained = first_filter_clear & np.isfinite(kt_prime)
    retained_f = pd.Series(retained.astype(float))
    min_retained = int(np.ceil(0.3 * 91))
    prev_count = retained_f.rolling(91, min_periods=91).sum().to_numpy()
    next_count = retained_f.iloc[::-1].rolling(91, min_periods=91).sum().iloc[::-1].to_numpy()
    enough_30pct = (prev_count >= min_retained) & (next_count >= min_retained)

    kt_prime_masked = np.where(retained, kt_prime, np.nan)
    KTp = pd.Series(kt_prime_masked).rolling(181, center=True, min_periods=1).std(ddof=0).to_numpy()
    # Keep second filter on the subset retained by first filter (k<0.3),
    # consistent with the textual method description ("remaining data").
    second_filter_clear = first_filter_clear & enough_30pct & np.isfinite(KTp) & (KTp < 0.02)
    cloud_flag[second_filter_clear] = 0.0

    bad = (
        np.isnan(ghi)
        | np.isnan(dhi)
        | np.isnan(ghi_extra)
        | np.isnan(zenith)
        | (ghi <= 0)
        | (dhi < 0)
    )
    cloud_flag[bad] = np.nan

    diagnostics = {"k": k, "kt": kt, "M": M, "kt_prime": kt_prime, "KTp": KTp}
    for key in diagnostics:
        # Writable copy: ``np.asarray`` can alias read-only buffers (e.g. pandas ``rolling`` output).
        arr = np.array(diagnostics[key], dtype=float, copy=True)
        arr[bad] = np.nan
        diagnostics[key] = arr
    return _csd_to_output(idx, cloud_flag, "lefevre", diagnostics, return_diagnostics)


def _optimise_alpha(meas, clear):
    """
    Closed-form RMSE-optimal scalar alpha: min_a sqrt(mean((y-a*x)^2)).
    闭式 RMSE 最优标量 alpha。

    Mathematically equivalent to MATLAB ``fminsearch`` on the same
    scalar-alpha RMSE objective.
    数学上等价于 MATLAB 对同一标量 RMSE 目标调用 ``fminsearch``。

    Parameters
    ----------
    meas : np.ndarray
        Measured irradiance (clear-period subset).
        实测辐照度（晴空子集）。
    clear : np.ndarray
        Clear-sky irradiance (clear-period subset).
        晴空辐照度（晴空子集）。

    Returns
    -------
    float or None
        Optimal alpha, or None if denominator is non-positive.
        最优 alpha，若分母非正则返回 None。
    """
    denom = np.nansum(clear * clear)
    if not np.isfinite(denom) or denom <= 0:
        return None
    alpha = np.nansum(meas * clear) / denom
    if not np.isfinite(alpha):
        return None
    return float(alpha)



def brightsun_csd(zenith, ghi, ghi_clear, dhi, dhi_clear,
                  times, return_diagnostics=False):
    """
    BrightSun2020CSDc clear-sky detection (tri-component).
    BrightSun2020CSDc 晴空检测（三分量） [1]_。

    MATLAB mapping:
    ``BrightSun2020CSDc(zen, ghi, ghics, dif, difcs, LST)``.
    MATLAB 变量映射：``zen -> zenith``, ``ghics -> ghi_clear``,
    ``dif -> dhi``, ``difcs -> dhi_clear``。

    The method proceeds in four stages matching the MATLAB routine:
    该方法按四个阶段执行（与 MATLAB 例程一致）：

    1. Initial Reno-style CSD guess for candidate clear periods.
       初始 Reno 风格 CSD 猜测。
    2. Daily clear-sky optimisation scales GHI, DHI, BNI
       clear-sky curves independently (alpha bounds
       ``[0.7, 1.5]`` for GHI/BNI, ``[0.3, 1.5]`` for DHI).
       每日晴空优化独立缩放 GHI、DHI、BNI 晴空曲线。
    3. Tri-component analysis on optimised curves.
       基于优化曲线的三分量分析。
    4. Cascaded duration filters (90/30/10-min).
       级联持续时间滤波器。

    Parameters
    ----------
    zenith : array-like
        Solar zenith angle. [degrees]
        太阳天顶角。[度]
    ghi : array-like
        Global horizontal irradiance. [W/m^2]
        水平总辐照度。[瓦/平方米]
    ghi_clear : array-like
        Clear-sky GHI. [W/m^2]
        晴空水平总辐照度。[瓦/平方米]
    dhi : array-like
        Diffuse horizontal irradiance. [W/m^2]
        水平散射辐照度。[瓦/平方米]
    dhi_clear : array-like
        Clear-sky DHI. [W/m^2]
        晴空水平散射辐照度。[瓦/平方米]
    times : array-like or pd.DatetimeIndex
        Time index (MATLAB ``LST`` equivalent).
        时间索引。
    return_diagnostics : bool, default False
        If True, include method diagnostics.
        若为 True，返回方法诊断量。

    Returns
    -------
    out : pd.DataFrame
        Columns: ``is_clearsky``, ``cloud_flag`` (duration-filtered),
        ``method``; diagnostics when requested.
        列：``is_clearsky``、``cloud_flag``（经滤波）、``method``。

    Raises
    ------
    ValueError
        When input lengths do not match.
        输入长度不一致时。

    References
    ----------
    .. [1] Bright, J. M., Sun, X., Gueymard, C. A., Acord, B.,
       Wang, P., & Engerer, N. A. (2020). Bright-Sun: A globally
       applicable 1-min irradiance clear-sky detection model.
       Renewable and Sustainable Energy Reviews, 121, 109706.
    """
    zenith = _as_1d_array(zenith, "zenith")
    ghi = _as_1d_array(ghi, "ghi", n=len(zenith))
    ghi_clear = _as_1d_array(
        ghi_clear, "ghi_clear", n=len(zenith))
    dhi = _as_1d_array(dhi, "dhi", n=len(zenith))
    dhi_clear = _as_1d_array(
        dhi_clear, "dhi_clear", n=len(zenith))
    idx = _resolve_index(times, len(zenith))
    n_output = len(zenith)

    # Cap DHI at GHI / 限制 DHI 不超过 GHI
    dhi = np.minimum(dhi, ghi)
    dhi_clear = np.minimum(dhi_clear, ghi_clear)

    # BNI from closure / 由闭合计算 BNI
    mu0 = np.cos(np.radians(zenith))
    bni = _safe_divide(ghi - dhi, mu0)
    bni_clear = _safe_divide(ghi_clear - dhi_clear, mu0)

    # NaN removal (MATLAB pattern) / 删除 NaN 行
    idxs_nan = (np.isnan(ghi) | np.isnan(bni)
                | np.isnan(dhi))
    not_nan = ~idxs_nan

    ghi_c = ghi[not_nan].copy()
    bni_c = bni[not_nan].copy()
    dhi_c = dhi[not_nan].copy()
    ghics_c = ghi_clear[not_nan].copy()
    bnics_c = bni_clear[not_nan].copy()
    dhics_c = dhi_clear[not_nan].copy()
    zen_c = zenith[not_nan].copy()
    n_clean = len(ghi_c)

    times_c = None
    if isinstance(idx, pd.DatetimeIndex):
        times_c = idx[not_nan]

    # ---- Stage 1: initial Reno guess (sigma_lim=0.1) ----
    # 阶段 1：初始 Reno 猜测
    csd_initial, _ = _reno_cloud_flag(
        ghi_c, ghics_c,
        mean_lim=75.0, max_lim=75.0,
        lower_L_lim=-5.0, upper_L_lim=10.0,
        sigma_lim=0.1, X_lim=8.0,
    )
    csd_initial = np.where(
        np.isfinite(csd_initial), csd_initial, 1.0)

    # ---- Stage 2: daily optimisation (GHI, DHI, BNI) ----
    # 阶段 2：每日优化
    opt_thres = 30.0
    upper_a = 1.5
    lower_a = 0.7
    lower_a_dhi = 0.3

    if times_c is not None and len(times_c) > 0:
        day_key = times_c.floor("D")
        for d in np.unique(day_key):
            ix = np.where(day_key == d)[0]
            csdd = csd_initial[ix].copy()
            csdd[(dhi_c[ix] < opt_thres)
                 | (ghi_c[ix] < opt_thres)] = 1.0
            cm = csdd == 0.0
            if np.sum(cm) <= 60:
                continue
            # GHI / GHI 优化
            a = _optimise_alpha(ghi_c[ix][cm],
                                ghics_c[ix][cm])
            if a is not None:
                ghics_c[ix] *= np.clip(a, lower_a, upper_a)
            # DHI / DHI 优化
            a = _optimise_alpha(dhi_c[ix][cm],
                                dhics_c[ix][cm])
            if a is not None:
                dhics_c[ix] *= np.clip(a, lower_a_dhi,
                                       upper_a)
            # BNI / BNI 优化
            a = _optimise_alpha(bni_c[ix][cm],
                                bnics_c[ix][cm])
            if a is not None:
                bnics_c[ix] *= np.clip(a, lower_a, upper_a)

    # ---- Stage 3: tri-component analysis ----
    # 阶段 3：三分量分析
    cloud_ghi = _brightsun_component_flag(
        ghi_c, ghics_c, zen_c, window=10, is_ghi=True)
    cloud_dhi = _brightsun_component_flag(
        dhi_c, dhics_c, zen_c, window=10, is_ghi=False)

    # BNI: zenith-dependent kcb threshold
    # BNI：随天顶角变化的 kcb 阈值
    z_ref = np.arange(30.0, 90.01, 0.01)
    kc_lims = np.flip(np.linspace(0.5, 0.9, len(z_ref)))
    inds = np.searchsorted(
        z_ref, np.clip(zen_c, z_ref[0], z_ref[-1]),
        side="left")
    inds = np.clip(inds, 0, len(z_ref) - 1)
    kc_lim_c = kc_lims[inds]
    kcb_c = _safe_divide(bni_c, bnics_c)
    cloud_bni = np.ones(n_clean, dtype=float)
    cloud_bni[kcb_c > kc_lim_c] = 0.0

    csd_overall = np.ones(n_clean, dtype=float)
    csd_overall[(cloud_ghi + cloud_dhi + cloud_bni) == 0] = 0.0

    # ---- Stage 4: duration filters ----
    # 阶段 4：持续时间滤波器
    # Inline centred Hankel filter: sum forward window, shift to centre
    # 内联居中 Hankel 滤波器：前向窗口求和后平移居中
    def _dur(flags, w, tol):
        h = np.nansum(_hankel_window(flags, w), axis=1)
        h = np.concatenate([np.full(w // 2, np.nan),
                            h[:n_clean - w // 2]])
        out = np.zeros(n_clean, dtype=float)
        out[h > tol] = 1.0
        return out

    # 1st: 90 min, tolerance 10 / 第一滤波器
    csd_1st = _dur(csd_overall, 90, 10)

    # Sunrise/sunset proximity (zen ≈ 85°)
    # 日出日落邻近
    ss_idx = np.where(np.round(zen_c) == 85)[0]
    if len(ss_idx) > 0:
        all_i = np.arange(n_clean)
        ss_s = np.sort(ss_idx)
        pos = np.searchsorted(ss_s, all_i)
        pos = np.clip(pos, 0, len(ss_s) - 1)
        pl = np.clip(pos - 1, 0, len(ss_s) - 1)
        dist_ss = np.minimum(
            np.abs(all_i - ss_s[pos]),
            np.abs(all_i - ss_s[pl]))
    else:
        dist_ss = np.full(n_clean, n_clean)
    csd_1st[dist_ss < 90] = 0.0

    # 2nd: 30 min, tolerance 0 / 第二滤波器
    csd_2nd = _dur(csd_overall, 30, 0)

    # 3rd: 10 min, tolerance 2 (sunrise/sunset override)
    # 第三滤波器（日出日落覆盖）
    csd_3rd = _dur(csd_overall, 10, 2)
    csd_2nd[(csd_3rd == 0) & (dist_ss < 90)] = 0.0

    # Final: all must agree clear / 最终合并
    csd_filtered = np.ones(n_clean, dtype=float)
    csd_filtered[
        (csd_overall + csd_1st + csd_2nd) == 0] = 0.0

    # ---- Map back to full length ----
    # 映射回完整长度
    cloud_flag = np.ones(n_output, dtype=float)
    cloud_flag[not_nan] = csd_filtered
    cloud_flag[idxs_nan] = np.nan

    def _expand(arr):
        full = np.full(n_output, np.nan)
        full[not_nan] = arr
        return full

    diagnostics = {
        "cloud_ghi": _expand(cloud_ghi),
        "cloud_dhi": _expand(cloud_dhi),
        "cloud_bni": _expand(cloud_bni),
        "kcb": _expand(kcb_c),
        "kc_lim": _expand(kc_lim_c),
        "mu0": mu0,
        "bni": bni,
        "bni_clear": bni_clear,
    }
    return _csd_to_output(
        idx, cloud_flag, "brightsun",
        diagnostics, return_diagnostics)


def detect_clearsky(method, **kwargs):
    """
    Wrapper for clear-sky detection methods.
    晴空检测方法封装函数。

    Parameters
    ----------
    method : {"reno", "ineichen", "lefevre", "brightsun"}
        Method name.
        方法名称。
    **kwargs : dict
        Arguments forwarded to method function.
        传递给具体方法函数的参数。

    Returns
    -------
    out : pd.DataFrame
        Standardized CSD output.
        标准化 CSD 输出。

    Raises
    ------
    ValueError
        When method is not one of reno, ineichen, lefevre, brightsun.
        方法名不在支持列表中时。
    """
    method_key = str(method).strip().lower()
    if method_key == "reno":
        return reno_csd(**kwargs)
    if method_key == "ineichen":
        return ineichen_csd(**kwargs)
    if method_key == "lefevre":
        return lefevre_csd(**kwargs)
    if method_key == "brightsun":
        return brightsun_csd(**kwargs)
    raise ValueError(
        "Unsupported method. Choose one of brightsun, ineichen, lefevre, reno. / "
        "不支持的方法，请选择 brightsun、ineichen、lefevre、reno。"
    )
