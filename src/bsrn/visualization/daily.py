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

from bsrn.constants import WONG_PALETTE

from bsrn.dataset import BSRNDataset

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

# LR_SPECS missing tokens for plot-only masking (does not alter :meth:`BSRNDataset.data`).
# LR_SPECS 缺失码，仅用于绘图前掩膜（不修改 :meth:`BSRNDataset.data`）。
_PLOT_MISSING_I4 = frozenset(
    {"ghi", "bni", "dhi", "lwd", "pressure", "swu", "lwu", "net"}
)
_PLOT_MISSING_F51 = frozenset({"temp", "rh"})
_PLOT_MISSING_LR4000_TEMPS = frozenset(
    {"dt1d", "dt2d", "dt3d", "btd", "dt1u", "dt2u", "dt3u", "btu"}
)


def _mask_bsrn_missing_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Replace BSRN archive sentinels with NaN for line/point layers only."""
    out = df.copy()
    for c in _PLOT_MISSING_I4:
        if c not in out.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        out[c] = col.mask(col == -999, np.nan)
    for c in _PLOT_MISSING_F51:
        if c not in out.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        out[c] = col.mask(np.isclose(col, -99.9, rtol=0, atol=0.05), np.nan)
    for c in _PLOT_MISSING_LR4000_TEMPS:
        if c not in out.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        out[c] = col.mask(np.isclose(col, -99.99, rtol=0, atol=0.01), np.nan)
    return out


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


def _build_qc_marker_frame(day_df, day_zenith):
    """
    Per-minute QC failures from existing ``flag*`` columns only (no ``run_qc``).

    仅根据数据框上已有的 ``flag*`` 列绘制失败点（不调用 ``run_qc``）。
    """
    df = day_df.copy()
    df["zenith"] = day_zenith.reindex(df.index).to_numpy()
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


def _load_month_archive_for_daily(file_path=None, df=None):
    """
    Load one month and add closure diagnostics (GHI vs sum).

    Requires ``zenith`` and ``apparent_zenith`` on the frame (from
    :meth:`~bsrn.dataset.BSRNDataset.solpos`). This module does not compute
    geometry, PPL masks, clear-sky, or QC.

    加载单月并加入闭合诊断量。帧上须已有 ``zenith``、``apparent_zenith``（``solpos()``）。
    本模块不计算几何、不做 PPL/晴空/QC。为绘图可读性，会将 LR 缺失码
    （``-999``、``-99.9``、``-99.99`` 等）在**副本**上替换为 NaN；不修改
    :meth:`~bsrn.dataset.BSRNDataset.data` 的缓存。
    """
    if df is not None:
        plot_df = df.copy()
    else:
        plot_df = BSRNDataset.from_file(file_path).data()

    plot_df = _mask_bsrn_missing_for_plot(plot_df)

    unique_months = np.unique(plot_df.index.to_period("M"))
    if len(unique_months) != 1:
        raise ValueError(
            f"Input file must contain exactly one month of data. "
            f"Found {len(unique_months)} months: {unique_months}"
        )

    if "zenith" not in plot_df.columns or "apparent_zenith" not in plot_df.columns:
        raise ValueError(
            "Daily plots require ``zenith`` and ``apparent_zenith`` on the DataFrame. "
            "Call :meth:`BSRNDataset.solpos` before plotting. / "
            "请先调用 ``solpos()``，使数据框包含 ``zenith`` 与 ``apparent_zenith``。"
        )
    zenith = plot_df["zenith"]
    apparent_zenith = plot_df["apparent_zenith"]
    mu0 = np.cos(np.radians(apparent_zenith))

    plot_df["sum_irrad"] = plot_df["bni"] * mu0 + plot_df["dhi"]
    plot_df["gh_ratio"] = plot_df["ghi"] / plot_df["sum_irrad"]
    plot_df["gh_diff"] = plot_df["ghi"] - plot_df["sum_irrad"]

    return plot_df.sort_index(), zenith


def _ggplot_bsrn_daily_one_day(day_df, day_zenith, title=None,
                                show_qc_markers=True):
    """
    Build the standard 3×3 facet times ggplot for one UTC day.

    Clear-sky ribbon/lines are drawn only for irradiance columns present on ``day_df``.
    QC markers are drawn only when ``show_qc_markers`` and matching ``flag*`` columns exist.
    晴空曲线仅当帧上存在对应 ``*_clear`` 列时绘制；QC 标记仅当存在相应 ``flag*`` 列时绘制。
    """
    measured_color = WONG_PALETTE[0]
    clearsky_color = WONG_PALETTE[1]
    ribbon_color = clearsky_color
    width_inch = 160 / 25.4
    height_inch = width_inch / 1.8

    day_df = day_df.copy()
    day_work = day_df.reset_index()
    _tcol = day_work.columns[0]
    if _tcol != "time":
        day_work = day_work.rename(columns={_tcol: "time"})

    main_vars = ["ghi", "bni", "dhi"]
    if "lwd" in day_work.columns:
        main_vars.append("lwd")
    clear_vars = [
        c
        for c in ("ghi_clear", "bni_clear", "dhi_clear", "lwd_clear")
        if c in day_work.columns
    ]

    day_main_measured = day_work.melt(
        id_vars=["time"],
        value_vars=main_vars,
        var_name="parameter",
        value_name="measured",
    )
    day_main_measured["parameter"] = day_main_measured["parameter"].str.upper()
    if clear_vars:
        day_main_clear = day_work.melt(
            id_vars=["time"],
            value_vars=clear_vars,
            var_name="parameter",
            value_name="clearsky",
        )
        day_main_clear["parameter"] = (
            day_main_clear["parameter"].str.replace("_clear", "").str.upper()
        )
        day_main = pd.merge(
            day_main_measured, day_main_clear, on=["time", "parameter"], how="left"
        )
    else:
        day_main = day_main_measured.copy()
        day_main["clearsky"] = np.nan

    day_df_diag = day_df.copy()
    day_df_diag.loc[day_zenith >= 90, "gh_ratio"] = np.nan

    diag_vars = [
        c
        for c in ("gh_diff", "gh_ratio", "temp", "rh", "pressure")
        if c in day_df_diag.columns
    ]
    diag_work = day_df_diag.reset_index()
    _t2 = diag_work.columns[0]
    if _t2 != "time":
        diag_work = diag_work.rename(columns={_t2: "time"})
    day_diag = diag_work.melt(
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
    if show_qc_markers:
        marker_df = _build_qc_marker_frame(day_df, day_zenith)
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

    layers = [ggplot(day_all, aes(x="time"))]
    if not cs_data.empty:
        if not cs_sw.empty:
            layers.append(
                geom_ribbon(
                    data=cs_sw,
                    mapping=aes(ymin=0, ymax="clearsky"),
                    fill=ribbon_color,
                    alpha=0.25,
                )
            )
        layers.append(
            geom_line(
                data=cs_data,
                mapping=aes(y="clearsky"),
                color=clearsky_color,
                size=0.3,
            )
        )
    layers.append(geom_line(aes(y="measured"), color=measured_color, size=0.3))
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


def plot_bsrn_daily_day(file_path, day, show_qc_markers=True,
                        output_file=None, title=None, df=None):
    """
    Plot one UTC day from a single-month BSRN archive (same layout as each booklet page).
    从单月 BSRN 存档绘制一个 UTC 日（版式与手册单页相同）。

    The frame (from ``df`` or loaded from ``file_path``) must already include
    ``zenith`` and ``apparent_zenith`` (e.g. after :meth:`BSRNDataset.solpos`).

    Parameters
    ----------
    file_path : str, optional
        Path to a single-month archive when ``df`` is not passed; otherwise ignored.
        未传 ``df`` 时的单月 ``.dat.gz`` 路径；传入 ``df`` 时可省略。
    day : str, datetime.date, datetime.datetime, or pd.Timestamp
        Calendar day to plot (UTC), e.g. ``"2024-01-15"`` or ``pd.Timestamp("2024-01-15")``.
        要绘制的日历日（UTC）。
    show_qc_markers : bool, default True
        If True, overlay QC failure markers where matching ``flag*`` columns exist
        (no QC is run here; use :meth:`~bsrn.dataset.BSRNDataset.qc_test` first).
        为 True 时在已有 ``flag*`` 列的分面上叠加失败点（此处不运行 QC；请先 ``qc_test()``）。
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
    plot_df, zenith = _load_month_archive_for_daily(file_path=file_path, df=df)

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
        src = file_path if file_path is not None else "<dataframe>"
        raise ValueError(
            f"No rows for UTC day {day_start.date()} in {src!r} "
            f"(index range {plot_df.index.min()} … {plot_df.index.max()})."
        )

    day_zenith = zenith.loc[day_df.index]

    p = _ggplot_bsrn_daily_one_day(
        day_df,
        day_zenith,
        title,
        show_qc_markers=show_qc_markers,
    )
    if output_file is not None:
        p.save(output_file, dpi=300)
    return p


def plot_bsrn_daily_booklet(file_path, output_file, show_qc_markers=True,
                            title=None, df=None):
    """
    Generate a multi-page PDF booklet where each day is one page from a BSRN archive file.
    从 BSRN 存档文件生成多页 PDF 手册，每一天占一页。

    The frame must include ``zenith`` and ``apparent_zenith`` (see :meth:`BSRNDataset.solpos`).

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (.dat.gz).
        BSRN 站点存档文件的路径 (.dat.gz)。
    output_file : str
        Path to the output PDF file.
        输出 PDF 文件的路径。
    show_qc_markers : bool, default True
        Draw QC markers per day when ``flag*`` columns are present.
        当存在 ``flag*`` 列时为每日绘制 QC 标记。
    title : str, optional
        Plot title on every page. If None (default), no title.
        每页图标题；默认 None 不显示。

    Returns
    -------
    None
        The function saves the plots to the specified PDF file.
        该函数将图表保存到指定的 PDF 文件中。
    """
    plot_df, zenith = _load_month_archive_for_daily(file_path=file_path, df=df)

    print(f"Generating PDF booklet: {output_file}...")
    with PdfPages(output_file) as pdf:
        for _, day_df in plot_df.groupby(plot_df.index.date):
            day_zenith = zenith.loc[day_df.index]
            p = _ggplot_bsrn_daily_one_day(
                day_df,
                day_zenith,
                title,
                show_qc_markers=show_qc_markers,
            )
            fig = p.draw()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Done.")
