"""
Visualization of irradiance separation: k vs kt scatter (Erbs, BRL, etc.).
辐照分离可视化：k vs kt 散点图（Erbs、BRL 等）。
"""

import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_point,
    theme_minimal, theme,
    element_text, element_blank, labs,
    scale_x_continuous, scale_y_continuous,
    scale_color_cmap, coord_fixed, facet_wrap
)

from bsrn.physics import geometry
from bsrn.utils.calculations import calc_kt
from bsrn.modeling import erbs_separation, brl_separation, engerer2_separation, yang4_separation

# Supported model names for k vs kt plot (Erbs, BRL, Engerer2, Yang4).
SUPPORTED_SEPARATION_MODELS = ("erbs", "brl", "engerer2", "yang4")


def plot_k_vs_kt(df, models, lat, lon, ghi_col="ghi", dhi_col="dhi", k_mod_cols=None,
                 output_file=None, title=None):
    """
    Faceted scatter plot of k (diffuse fraction) vs kt (clearness index) from a DataFrame.
    根据 DataFrame 绘制 k vs kt 散点图，按模型分面。

    One panel per model; measured (gray) and model (colored) in each panel.
    每个模型一个面板；各面板内为实测（灰）与模型（着色）点。

    Parameters
    ----------
    df : pd.DataFrame
        Input data with DatetimeIndex and at least ``ghi_col`` and ``dhi_col``.
        输入数据，需含 DatetimeIndex 及 ghi、dhi 列。
    models : sequence of str
        Model names to plot, e.g. ``('erbs', 'brl')``. Each in SUPPORTED_SEPARATION_MODELS.
        要绘制的模型名，如 ``('erbs', 'brl')``。
    lat : float
        Latitude. [degrees] 纬度。[度]
    lon : float
        Longitude. [degrees] 经度。[度]
    ghi_col : str, default "ghi"
        Column name for GHI. GHI 列名。
    dhi_col : str, default "dhi"
        Column name for DHI. DHI 列名。
    k_mod_cols : dict or None, optional
        Map model name to column name for model k. If None, runs each separation model.
        For ``engerer2`` and ``yang4``, k must be pre-computed and provided here (both need clear-sky GHI).
        模型名到 k 列名的映射。engerer2、yang4 的 k 须预先计算并通过此处传入。
    output_file : str, optional
        Path to save the figure. 保存路径。
    title : str, optional
        Overall plot title. 图标题。

    Notes
    -----
    To include Engerer2 or Yang4 (both require clear-sky GHI): add clear-sky GHI to ``df``
    (e.g. :func:`bsrn.physics.clearsky.add_clearsky_columns`), run the separation with
    ``ghi_clear=df["ghi_clear"].values``, assign the result's ``"k"`` to a column, then pass
    ``k_mod_cols={"engerer2": "k_engerer2", "yang4": "k_yang4"}``. 要绘制 Engerer2/Yang4：
    先在 df 中加入晴空 GHI，调用对应 separation(..., ghi_clear=...)，将 k 写入列再通过 k_mod_cols 传入。

    Returns
    -------
    p : plotnine.ggplot
        The ggplot object. 返回的 ggplot 对象。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex. / df 须为 DatetimeIndex。")
    for col, name in [(ghi_col, "ghi"), (dhi_col, "dhi")]:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing column '{col}' ({name}). / 缺少列 '{col}'。")

    models = tuple(m.strip().lower() for m in models)
    if not models:
        raise ValueError("models must be a non-empty sequence. / models 不可为空。")
    for m in models:
        if m not in SUPPORTED_SEPARATION_MODELS:
            raise ValueError(
                f"model must be one of {SUPPORTED_SEPARATION_MODELS}, got {m!r}. / "
                f"model 必须为 {SUPPORTED_SEPARATION_MODELS} 之一。"
            )

    use = df[[ghi_col, dhi_col]].dropna()
    if len(use) == 0:
        raise ValueError("No valid ghi/dhi rows after dropna. / dropna 后无有效 ghi/dhi 行。")
    times = use.index
    ghi = np.asarray(use[ghi_col], dtype=float)
    dhi = np.asarray(use[dhi_col], dtype=float)
    k_meas = np.clip(np.where(ghi > 0, dhi / ghi, np.nan), 0.0, 1.0)

    zenith = np.asarray(
        geometry.get_solar_position(times, lat, lon)["zenith"], dtype=float
    )
    bni_extra = np.asarray(geometry.get_bni_extra(times), dtype=float)
    kt = calc_kt(ghi, zenith, bni_extra, min_mu0=0.065, max_clearness_index=1.0)

    day = zenith < 87
    kt = kt[day]
    k_meas = k_meas[day]
    times_day = times[day]
    ghi_day = ghi[day]

    n = len(kt)
    k_mod_cols = k_mod_cols or {}
    rows = []
    for model in models:
        if model in k_mod_cols and k_mod_cols[model] in df.columns:
            k_mod = np.asarray(df.loc[use.index, k_mod_cols[model]], dtype=float)[day]
        else:
            if model == "erbs":
                result = erbs_separation(times_day, ghi_day, lat, lon)
            elif model == "brl":
                result = brl_separation(times_day, ghi_day, lat, lon)
            elif model == "engerer2":
                raise ValueError(
                    "engerer2 requires pre-computed k. Run engerer2_separation and pass column in k_mod_cols. / "
                    "engerer2 需预先计算 k，请运行 engerer2_separation 并通过 k_mod_cols 传入列名。"
                )
            elif model == "yang4":
                raise ValueError(
                    "yang4 requires pre-computed k. Run yang4_separation and pass column in k_mod_cols. / "
                    "yang4 需预先计算 k，请运行 yang4_separation 并通过 k_mod_cols 传入列名。"
                )
            else:
                raise ValueError(f"Unknown model {model!r}. / 未知模型 {model!r}。")
            k_mod = np.asarray(result["k"], dtype=float)
        for i in range(n):
            rows.append({"kt": kt[i], "k": k_meas[i], "model": model.title(), "source": "measured"})
            rows.append({"kt": kt[i], "k": k_mod[i], "model": model.title(), "source": "model"})
    plot_df = pd.DataFrame(rows)

    return _plot_k_vs_kt_facet(
        plot_df, output_file=output_file, title=title
    )


def _get_density(x, y, n=200):
    """
    Grid-based 2-D density estimate, equivalent to R's MASS::kde2d + findInterval lookup.
    基于网格的二维密度估计，等同于 R 的 MASS::kde2d + findInterval。
    """
    from scipy.ndimage import gaussian_filter

    finite = np.isfinite(x) & np.isfinite(y)
    density = np.full(len(x), np.nan)
    if finite.sum() < 2:
        return density
    xf, yf = x[finite], y[finite]
    counts, xedges, yedges = np.histogram2d(xf, yf, bins=n)
    z = gaussian_filter(counts.T, sigma=n / 25.0)
    ix = np.clip(np.digitize(xf, xedges) - 1, 0, n - 1)
    iy = np.clip(np.digitize(yf, yedges) - 1, 0, n - 1)
    density[finite] = z[iy, ix]
    return density


def _plot_k_vs_kt_facet(plot_df, output_file=None, title=None):
    """Draw faceted k vs kt scatter (measured gray, model colored by density) per model."""
    df_meas = plot_df[plot_df["source"] == "measured"].dropna().copy()
    df_mod = plot_df[plot_df["source"] == "model"].dropna().copy()
    df_mod = df_mod[df_mod["k"] < 1].copy()

    # Density per model on n=200 grid (like R MASS::kde2d)
    parts = []
    for _, grp in df_mod.groupby("model"):
        grp = grp.copy()
        grp["density"] = _get_density(grp["kt"].values, grp["k"].values, n=200)
        parts.append(grp)
    df_mod = pd.concat(parts).sort_values("density")

    n_facets = plot_df["model"].nunique()
    panel_width_inch = 80 / 25.4
    width_inch = panel_width_inch * n_facets
    fig_h = panel_width_inch * 0.9

    p = (
        ggplot()
        + geom_point(
            data=df_meas,
            mapping=aes(x="kt", y="k"),
            color="gray",
            alpha=0.15,
            size=0.3,
            stroke=0,
            raster=True
        )
        + geom_point(
            data=df_mod,
            mapping=aes(x="kt", y="k", color="density"),
            alpha=0.5,
            size=0.5,
            stroke=0,
            raster=True
        )
        + facet_wrap("model", ncol=n_facets, scales="free")
        + scale_color_cmap(cmap_name="viridis", name="density")
        + labs(
            x=r"$k_t$ (clearness index)",
            y=r"$k$ (diffuse fraction)"
        )
        + scale_x_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.2))
        + scale_y_continuous(limits=(0, 1), breaks=np.arange(0, 1.1, 0.2))
        + coord_fixed(ratio=1)
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=7),
            plot_title=element_text(size=7),
            axis_title=element_text(size=7),
            axis_text=element_text(size=7),
            legend_position="none",
            panel_grid_minor=element_blank(),
            figure_size=(width_inch, fig_h)
        )
    )
    if output_file:
        p.save(output_file, dpi=300)
    return p
