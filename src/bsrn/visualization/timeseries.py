import pandas as pd
import numpy as np
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    geom_hline,
    geom_ribbon,
    facet_wrap,
    theme,
    element_text,
    theme_minimal,
    labs,
    scale_x_datetime,
    scale_color_manual,
    scale_shape_manual,
    guides,
    guide_legend,
    element_line,
)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from bsrn.constants import BSRN_STATIONS, WONG_PALETTE
from bsrn.physics.geometry import get_solar_position, get_bni_extra
from bsrn.qc.ppl import ghi_ppl_test, bni_ppl_test, dhi_ppl_test


from bsrn.io.reader import read_station_to_archive
from bsrn.modeling.clear_sky import add_clearsky_columns
from bsrn.qc.wrapper import run_qc

# L1–L6 QC markers: WONG[2–6] for L1–5, black for L6 (WONG[0–1] reserved for line series).
# L1–L6 标记：L1–5 用 Wong 剩余五色，L6 黑色；WONG[0–1] 留给实测/晴空线。
_QC_MARKER_BLACK = "#000000"
_QC_LEVEL_COLORS = {
    "L1": WONG_PALETTE[2],
    "L2": WONG_PALETTE[3],
    "L3": WONG_PALETTE[4],
    "L4": WONG_PALETTE[5],
    "L5": WONG_PALETTE[6],
    "L6": _QC_MARKER_BLACK,
}
_QC_LEVEL_SHAPES = {
    "L1": "o",
    "L2": "^",
    "L3": "s",
    "L4": "D",
    "L5": "v",
    "L6": "p",
}
# Plotted markers stay small; legend keys use 2× size for readability.
# 图上点保持小号；图例符号为 2 倍便于辨认。
_QC_MARKER_SIZE = 0.8
_QC_MARKER_STROKE = 0.12
_QC_MARKER_LEGEND_SIZE = _QC_MARKER_SIZE * 2
_QC_MARKER_LEGEND_STROKE = _QC_MARKER_STROKE * 2


def _qc_marker_slices(df, mask, ycol, param_facet, level, chunks):
    """Append marker rows for timestamps where mask is True and y is finite."""
    if ycol not in df.columns or mask is None or not mask.any():
        return
    m = mask.fillna(False) & df[ycol].notna()
    if not m.any():
        return
    sub = df.loc[m]
    chunks.append(
        pd.DataFrame(
            {
                "time": sub.index,
                "parameter": param_facet,
                "qc_y": sub[ycol].to_numpy(),
                "qc_level": f"L{level}",
            }
        )
    )


def _build_qc_marker_frame(day_df, day_zenith, station_code):
    """
    Per-minute QC failures for faceted timeseries (Levels 1–6; Wong [2–6] + black, distinct shapes).
    分面时间序列用的逐分钟 QC 失败点（1–6 级；Wong 五色 + 黑，各级不同形状）。
    """
    if station_code is None:
        return None
    df = day_df.copy()
    df["zenith"] = day_zenith.reindex(df.index).to_numpy()
    df = run_qc(df, station_code=station_code)
    chunks = []

    pairs_l1 = [
        ("flagPPLGHI", "ghi", "GHI"),
        ("flagPPLBNI", "bni", "BNI"),
        ("flagPPLDHI", "dhi", "DHI"),
        ("flagPPLLWD", "lwd", "LWD"),
    ]
    for flg, ycol, pan in pairs_l1:
        if flg in df.columns:
            _qc_marker_slices(df, df[flg] == 1, ycol, pan, 1, chunks)

    pairs_l2 = [
        ("flagERLGHI", "ghi", "GHI"),
        ("flagERLBNI", "bni", "BNI"),
        ("flagERLDHI", "dhi", "DHI"),
        ("flagERLLWD", "lwd", "LWD"),
    ]
    for flg, ycol, pan in pairs_l2:
        if flg in df.columns:
            _qc_marker_slices(df, df[flg] == 1, ycol, pan, 2, chunks)

    if "flag3lowSZA" in df.columns and "flag3highSZA" in df.columns:
        m3 = (df["flag3lowSZA"] == 1) | (df["flag3highSZA"] == 1)
        for ycol, pan in (("ghi", "GHI"), ("bni", "BNI"), ("dhi", "DHI")):
            _qc_marker_slices(df, m3, ycol, pan, 3, chunks)
        _qc_marker_slices(df, m3, "gh_diff", "GHI-SUM", 3, chunks)

    if all(c in df.columns for c in ("flagKKt", "flagKlowSZA", "flagKhighSZA")):
        m4 = (df["flagKKt"] == 1) | (df["flagKlowSZA"] == 1) | (df["flagKhighSZA"] == 1)
        _qc_marker_slices(df, m4, "ghi", "GHI", 4, chunks)
        _qc_marker_slices(df, m4, "dhi", "DHI", 4, chunks)
        z = df["zenith"] if "zenith" in df.columns else None
        if z is not None:
            gr = df["gh_ratio"].where(z < 90)
        else:
            gr = df["gh_ratio"]
        tmp = df.copy()
        tmp["_ghr_plot"] = gr
        _qc_marker_slices(tmp, m4, "_ghr_plot", "GHI/SUM", 4, chunks)

    if all(c in df.columns for c in ("flagKbKt", "flagKb", "flagKt")):
        m5 = (df["flagKbKt"] == 1) | (df["flagKb"] == 1) | (df["flagKt"] == 1)
        _qc_marker_slices(df, m5, "ghi", "GHI", 5, chunks)
        _qc_marker_slices(df, m5, "bni", "BNI", 5, chunks)

    if "flagTracker" in df.columns:
        _qc_marker_slices(df, df["flagTracker"] == 1, "ghi", "GHI", 6, chunks)
        _qc_marker_slices(df, df["flagTracker"] == 1, "bni", "BNI", 6, chunks)

    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True)
    if out.empty:
        return None
    cat_lvls = [f"L{i}" for i in range(1, 7)]
    out["qc_level"] = pd.Categorical(out["qc_level"], categories=cat_lvls, ordered=True)
    return out


def _load_month_archive_for_timeseries(file_path, station_code, apply_qc):
    """
    Load one month from archive, add geometry, optional clearsky and PPL masking.
    加载单月存档，加入几何、可选晴空与 PPL 掩膜。
    """
    plot_df = read_station_to_archive(file_path)
    if plot_df is None:
        raise ValueError(f"Failed to read BSRN file: {file_path}")

    unique_months = np.unique(plot_df.index.to_period("M"))
    if len(unique_months) != 1:
        raise ValueError(
            f"Input file must contain exactly one month of data. "
            f"Found {len(unique_months)} months: {unique_months}"
        )

    stn = BSRN_STATIONS.get(station_code, {"lat": 0, "lon": 0, "elev": 0})
    solpos = get_solar_position(plot_df.index, stn["lat"], stn["lon"], stn["elev"])
    mu0 = np.cos(np.radians(solpos["apparent_zenith"]))
    zenith = solpos["zenith"]

    plot_df["sum_irrad"] = plot_df["bni"] * mu0 + plot_df["dhi"]
    plot_df["gh_ratio"] = plot_df["ghi"] / plot_df["sum_irrad"]
    plot_df["gh_diff"] = plot_df["ghi"] - plot_df["sum_irrad"]

    if station_code is not None:
        plot_df = add_clearsky_columns(plot_df, station_code)

    if apply_qc:
        bni_extra = get_bni_extra(plot_df.index)
        ghi_mask = ghi_ppl_test(plot_df["ghi"], solpos["apparent_zenith"], bni_extra)
        bni_mask = bni_ppl_test(plot_df["bni"], bni_extra)
        dhi_mask = dhi_ppl_test(plot_df["dhi"], solpos["apparent_zenith"], bni_extra)
        plot_df.loc[~ghi_mask, "ghi"] = np.nan
        plot_df.loc[~bni_mask, "bni"] = np.nan
        plot_df.loc[~dhi_mask, "dhi"] = np.nan

    return plot_df.sort_index(), zenith


def _ggplot_bsrn_timeseries_one_day(
    day_df, day_zenith, title=None, station_code=None, show_qc_markers=True
):
    """
    Build the standard 3×3 facet times ggplot for one UTC day.
    构建单日 UTC 的 3×3 分面时间序列图。
    """
    measured_color = WONG_PALETTE[0]
    clearsky_color = WONG_PALETTE[1]
    ribbon_color = clearsky_color
    width_inch = 160 / 25.4
    height_inch = width_inch / 1.8

    main_vars = ["ghi", "bni", "dhi"]
    if "lwd" in day_df.columns:
        main_vars.append("lwd")
    clear_vars = ["ghi_clear", "bni_clear", "dhi_clear"]
    if "lwd_clear" in day_df.columns:
        clear_vars.append("lwd_clear")

    day_main_measured = day_df.reset_index().melt(
        id_vars=["time"],
        value_vars=main_vars,
        var_name="parameter",
        value_name="measured",
    )
    day_main_clear = day_df.reset_index().melt(
        id_vars=["time"],
        value_vars=clear_vars,
        var_name="parameter",
        value_name="clearsky",
    )
    day_main_measured["parameter"] = day_main_measured["parameter"].str.upper()
    day_main_clear["parameter"] = (
        day_main_clear["parameter"].str.replace("_clear", "").str.upper()
    )
    day_main = pd.merge(
        day_main_measured, day_main_clear, on=["time", "parameter"], how="left"
    )

    day_df_diag = day_df.copy()
    day_df_diag.loc[day_zenith >= 90, "gh_ratio"] = np.nan

    diag_vars = ["gh_diff", "gh_ratio", "temp", "rh", "pressure"]
    day_diag = day_df_diag.reset_index().melt(
        id_vars=["time"],
        value_vars=diag_vars,
        var_name="parameter",
        value_name="value",
    )
    name_map = {
        "gh_diff": "GHI-SUM",
        "gh_ratio": "GHI/SUM",
        "temp": "TMP",
        "rh": "RH",
        "pressure": "SP",
    }
    day_diag["parameter"] = day_diag["parameter"].map(name_map)

    all_params = ["GHI", "BNI", "DHI", "GHI-SUM", "GHI/SUM", "LWD", "TMP", "RH", "SP"]
    cat_type = pd.CategoricalDtype(categories=all_params, ordered=True)

    day_diag_renamed = day_diag.rename(columns={"value": "measured"})
    day_diag_renamed["clearsky"] = np.nan
    day_all = pd.concat([day_main, day_diag_renamed], ignore_index=True)
    day_all["parameter"] = day_all["parameter"].astype(cat_type)

    marker_df = None
    if show_qc_markers and station_code:
        marker_df = _build_qc_marker_frame(day_df, day_zenith, station_code)
        if marker_df is not None:
            marker_df = marker_df.copy()
            marker_df["parameter"] = marker_df["parameter"].astype(cat_type)

    day_thresh = pd.DataFrame(
        {
            "time": day_df.index,
            "upper": np.where(day_zenith < 75, 1.08, 1.15),
            "lower": np.where(day_zenith < 75, 0.92, 0.85),
            "parameter": "GHI/SUM",
        },
        index=day_df.index,
    )
    day_thresh["parameter"] = day_thresh["parameter"].astype(cat_type)
    day_thresh.loc[day_zenith >= 90, ["upper", "lower"]] = np.nan

    hline_df = pd.DataFrame({"parameter": ["GHI/SUM"], "y": [1.0]})
    hline_df["parameter"] = hline_df["parameter"].astype(cat_type)

    cs_data = day_all.dropna(subset=["clearsky"])
    cs_sw = cs_data[cs_data["parameter"].isin(["GHI", "BNI", "DHI"])]

    qc_levels = [f"L{i}" for i in range(1, 7)]
    legend_pos = "bottom" if marker_df is not None and not marker_df.empty else "none"
    if legend_pos == "bottom":
        height_inch = height_inch + 0.35

    layers = [
        ggplot(day_all, aes(x="time")),
        geom_ribbon(
            data=cs_sw,
            mapping=aes(ymin=0, ymax="clearsky"),
            fill=ribbon_color,
            alpha=0.25,
        ),
        geom_line(
            data=cs_data, mapping=aes(y="clearsky"), color=clearsky_color, size=0.3
        ),
        geom_line(aes(y="measured"), color=measured_color, size=0.3),
    ]
    if marker_df is not None and not marker_df.empty:
        layers.append(
            geom_point(
                data=marker_df,
                mapping=aes(x="time", y="qc_y", color="qc_level", shape="qc_level"),
                size=_QC_MARKER_SIZE,
                stroke=_QC_MARKER_STROKE,
                na_rm=True,
            )
        )
        layers.append(
            scale_color_manual(
                name="QC level",
                breaks=qc_levels,
                values=[_QC_LEVEL_COLORS[k] for k in qc_levels],
            )
        )
        layers.append(
            scale_shape_manual(
                name="QC level",
                breaks=qc_levels,
                values=[_QC_LEVEL_SHAPES[k] for k in qc_levels],
            )
        )
        layers.append(
            guides(
                color=guide_legend(nrow=1),
                shape=guide_legend(
                    nrow=1,
                    override_aes={
                        "size": _QC_MARKER_LEGEND_SIZE,
                        "stroke": _QC_MARKER_LEGEND_STROKE,
                    },
                ),
            )
        )
    layers.extend(
        [
            geom_line(
                data=day_thresh,
                mapping=aes(y="upper"),
                color="#999999",
                size=0.3,
                linetype="dashed",
            ),
            geom_line(
                data=day_thresh,
                mapping=aes(y="lower"),
                color="#999999",
                size=0.3,
                linetype="dashed",
            ),
            geom_hline(
                data=hline_df, mapping=aes(yintercept="y"), color="#999999", size=0.2
            ),
            facet_wrap("~parameter", nrow=3, ncol=3, scales="free_y", drop=False),
            labs(
                **(
                    {"x": "Time (UTC)", "y": "Value", "title": title}
                    if title is not None
                    else {"x": "Time (UTC)", "y": "Value"}
                )
            ),
            theme_minimal(),
            theme(
                **(
                    {
                        "text": element_text(
                            family="Times New Roman", size=9, face="plain"
                        ),
                        "axis_title": element_text(size=9, face="plain"),
                        "axis_text": element_text(size=9, face="plain"),
                        "plot_title": element_text(
                            size=9, face="plain", margin={"b": 10}
                        ),
                        "legend_position": legend_pos,
                        "legend_title": element_text(size=9, face="plain"),
                        "legend_text": element_text(size=9, face="plain"),
                        "figure_size": (width_inch, height_inch),
                        "panel_grid_minor": element_line(alpha=0),
                        "strip_text": element_text(size=9, face="plain"),
                        "axis_text_x": element_text(
                            rotation=0, size=9, face="plain"
                        ),
                    }
                    | (
                        {
                            # Tight legend under facets, but leave room so it does not cover x title.
                            # 图例靠上但不压住「Time (UTC)」：不用大负 margin，x 轴标题保留下方空隙。
                            "legend_margin": 0,
                            "legend_box_spacing": 0,
                            "axis_title_x": element_text(
                                size=9,
                                face="plain",
                                margin={"t": 4, "b": 8},
                            ),
                        }
                        if legend_pos == "bottom"
                        else {}
                    )
                )
            ),
            scale_x_datetime(date_labels="%H:%M"),
        ]
    )
    return sum(layers[1:], layers[0])


def plot_bsrn_timeseries_day(
    file_path,
    day,
    station_code=None,
    apply_qc=False,
    show_qc_markers=True,
    output_file=None,
    title=None,
):
    """
    Plot one UTC day from a single-month BSRN archive (same layout as each booklet page).
    从单月 BSRN 存档绘制一个 UTC 日（版式与手册单页相同）。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (``.dat.gz``).
        BSRN 站点存档路径（``.dat.gz``）。
    day : str, datetime.date, datetime.datetime, or pd.Timestamp
        Calendar day to plot (UTC), e.g. ``"2024-01-15"`` or ``pd.Timestamp("2024-01-15")``.
        要绘制的日历日（UTC）。
    station_code : str, optional
        Station abbreviation for title and clearsky / QC (same as booklet).
        站点缩写，用于标题与晴空 / QC（与手册一致）。
    apply_qc : bool, default False
        If True, mask GHI/BNI/DHI with physically possible limits before plotting.
        若为 True，绘图前用物理可能限值掩膜 GHI/BNI/DHI。
    show_qc_markers : bool, default True
        If True and ``station_code`` is set, overlay Level 1–6 QC failure markers
        (Wong palette) on the relevant facets.
        若为 True 且提供了 ``station_code``，在相应分面上叠加 1–6 级 QC 失败标记（Wong 色）。
    output_file : str, optional
        If set, save the figure (e.g. ``".pdf"`` or ``".png"``) via plotnine.
        若设置则保存图形（如 ``.pdf`` / ``.png``）。
    title : str, optional
        Plot title. If None (default), no title is drawn.
        图标题；默认 None 不显示标题。

    Returns
    -------
    p : plotnine.ggplot.ggplot
        The figure; display in notebooks with the last expression or call ``.draw()``.
        图形对象；笔记本中作为最后一行显示或调用 ``.draw()``。
    """
    plot_df, zenith = _load_month_archive_for_timeseries(
        file_path, station_code, apply_qc
    )

    idx_tz = getattr(plot_df.index, "tz", None)
    day_anchor = pd.Timestamp(day)
    if idx_tz is not None:
        if day_anchor.tzinfo is None:
            day_anchor = day_anchor.tz_localize(idx_tz)
        else:
            day_anchor = day_anchor.tz_convert(idx_tz)
    day_start = day_anchor.floor("D")
    day_end = day_start + pd.Timedelta(days=1)
    mask = (plot_df.index >= day_start) & (plot_df.index < day_end)
    day_df = plot_df.loc[mask]
    if day_df.empty:
        raise ValueError(
            f"No rows for UTC day {day_start.date()} in {file_path!r} "
            f"(index range {plot_df.index.min()} … {plot_df.index.max()})."
        )

    day_zenith = zenith.loc[day_df.index]

    p = _ggplot_bsrn_timeseries_one_day(
        day_df,
        day_zenith,
        title,
        station_code=station_code,
        show_qc_markers=show_qc_markers,
    )
    if output_file is not None:
        p.save(output_file, dpi=300)
    return p


def plot_bsrn_timeseries_booklet(
    file_path,
    output_file,
    station_code=None,
    apply_qc=False,
    show_qc_markers=True,
    title=None,
):
    """
    Generate a multi-page PDF booklet where each day is one page from a BSRN archive file.
    从 BSRN 存档文件生成多页 PDF 手册，每一天占一页。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (.dat.gz).
        BSRN 站点存档文件的路径 (.dat.gz)。
    output_file : str
        Path to the output PDF file.
        输出 PDF 文件的路径。
    station_code : str, optional
        Station abbreviation used for Title and QC calculations.
        站点缩写，用于标题和质量控制计算。
    apply_qc : bool, default False
        If True, applies physically possible limits (PPL) and masks bad data.
        如果为 True，则应用物理可能极限 (PPL) 并屏蔽错误数据。
    show_qc_markers : bool, default True
        If True and ``station_code`` is set, draw Level 1–6 QC failure markers per day.
        若为 True 且设置了 ``station_code``，为每日绘制 1–6 级 QC 失败标记。
    title : str, optional
        Plot title on every page. If None (default), no title.
        每页图标题；默认 None 不显示。

    Returns
    -------
    None
        The function saves the plots to the specified PDF file.
        该函数将图表保存到指定的 PDF 文件中。
    """
    plot_df, zenith = _load_month_archive_for_timeseries(
        file_path, station_code, apply_qc
    )

    print(f"Generating PDF booklet: {output_file}...")
    with PdfPages(output_file) as pdf:
        for _, day_df in plot_df.groupby(plot_df.index.date):
            day_zenith = zenith.loc[day_df.index]
            p = _ggplot_bsrn_timeseries_one_day(
                day_df,
                day_zenith,
                title,
                station_code=station_code,
                show_qc_markers=show_qc_markers,
            )
            fig = p.draw()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Done.")
