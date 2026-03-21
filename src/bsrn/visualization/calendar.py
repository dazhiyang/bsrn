"""
Calendar-style irradiance plots.
日历样式的辐照度对比图。
"""

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_text,
    facet_grid,
    geom_line,
    geom_ribbon,
    geom_vline,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
    scale_color_manual,
    scale_fill_manual,
)


def plot_calendar(df, output_file, station_code, meas_col=None, clear_col=None, 
                  other_cols=None, labels=None, title=None):
    """
    Plot a one-page calendar-style comparison for multiple irradiance series (up to 7).
    生成单页日历样式对比图，展示多条辐照度时段序列（最多 7 条）。

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with UTC DatetimeIndex and 'zenith' column.
        已处理的 DataFrame，含 UTC DatetimeIndex 与 ``zenith`` 列。
    output_file : str
        Path to save the output PDF.
        保存 PDF 的路径。
    station_code : str
        BSRN station code for the title.
        用于标题的原站 BSRN 代码。
    meas_col : str, optional
        Column name for measured data (plotted as solid line).
        实测数据的列名（用实线绘制）。
    clear_col : str, optional
        Column name for clear-sky data (plotted with line and ribbon).
        晴空数据的列名（绘制线条与色带）。
    other_cols : list of str, optional
        Additional column names to plot as solid lines.
        要作为实线绘制的其他额外列名。
    labels : list of str, optional
        Labels for the legend corresponding to provided columns in order:
        [meas_col, clear_col] + other_cols. If None, column names are used.
        与 [实测, 晴空] + 其他列 对应的图例标签列表；为 None 时使用列名。
    title : str, optional
        Custom plot title. If None, an automatic title is generated.
        自定义标题；为 None 时自动生成。
    """
    if "zenith" not in df.columns:
        raise ValueError("df must contain a 'zenith' column. / df 必须包含 ``zenith`` 列。")

    columns = []
    if meas_col:
        columns.append(meas_col)
    if clear_col:
        columns.append(clear_col)
    if other_cols:
        columns.extend(other_cols)
        
    if not columns:
        raise ValueError("At least one column must be specified to plot. / 必须指定至少一列进行绘制。")

    if len(columns) > 7:
        import warnings
        warnings.warn("WONG_PALETTE has only 7 colors; first 7 will have distinct colors.")

    if labels is None:
        labels = columns

    df = df.sort_index()
    # For ceiling-aligned data, the last label might be the 1st of the next month.
    clean_idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    unique_months = clean_idx.shift(-1, freq="s").to_period("M").unique()
    if len(unique_months) != 1:
        raise ValueError(f"Found multiple months based on interval coverage: {list(unique_months)}")

    # Keep only daytime samples (sun above horizon + twilight).
    df_day = df.loc[df["zenith"] < 93.0].copy()
    if df_day.empty:
        raise ValueError("No daytime data found in input DataFrame.")

    # Calendar coordinates.
    times = df_day.index
    dates = times.normalize()
    first_date = dates.min()
    first_monday = first_date - pd.Timedelta(days=int(first_date.weekday()))
    week = ((dates - first_monday).days // 7).astype(int)

    day_series = pd.Series(dates, index=times)
    t_index = day_series.groupby(day_series).cumcount().to_numpy()

    base = pd.DataFrame(
        {
            "t_index": t_index,
            "date": dates,
            "weekday": dates.weekday,
            "week": np.clip(week, 0, 4),
        },
        index=times,
    )
    for col in columns:
        base[col] = df_day[col].to_numpy(dtype=float)

    # Melt for plotnine
    long_df = base.melt(
        id_vars=["t_index", "date", "weekday", "week"],
        value_vars=columns,
        var_name="kind",
        value_name="value",
    )

    # Automatic color mapping from WONG_PALETTE (Ordered)
    from bsrn.constants import WONG_PALETTE
    label_map = dict(zip(columns, labels))
    
    # Pre-define colors to ensure consistency
    colors_pool = WONG_PALETTE
    full_color_map = {}
    
    # Map colors based on provided columns to maintain specific color for meas/clear
    # This ensures meas_col is always Wong[0] if present, clear_col is Wong[1], etc.
    current_idx = 0
    if meas_col:
        full_color_map[label_map[meas_col]] = colors_pool[0]
        current_idx = 1
    if clear_col:
        full_color_map[label_map[clear_col]] = colors_pool[1]
        # Skip Wong[1] for other_cols if clear_col is already assigned
        if current_idx < 2:
            current_idx = 2
    
    if other_cols:
        for oc in other_cols:
            full_color_map[label_map[oc]] = colors_pool[current_idx % 7]
            current_idx += 1

    long_df["kind_label"] = pd.Categorical(
        [label_map[c] for c in long_df["kind"]], categories=labels, ordered=True
    )

    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    long_df["weekday"] = pd.Categorical(
        [weekday_labels[i] for i in long_df["weekday"]],
        categories=weekday_labels,
        ordered=True,
    )

    bounds = (
        base.reset_index()
        .groupby("date", as_index=False)
        .agg(
            t_start=("t_index", "min"),
            t_end=("t_index", "max"),
            week=("week", "first"),
            weekday=("weekday", "first"),
        )
    )
    bounds["weekday"] = pd.Categorical(
        [weekday_labels[i] for i in bounds["weekday"]],
        categories=weekday_labels,
        ordered=True,
    )

    if title is None:
        month_str = unique_months[0].strftime("%Y %b")
        title = f"{station_code} – {month_str} Comparison"

    width_inch = 160 / 25.4
    height_inch = width_inch * (5 / 7)
    max_t = int(base["t_index"].max())

    p = (
        ggplot(long_df, aes(x="t_index", y="value", color="kind_label", group="kind_label"))
        + geom_line(size=0.3)
        + geom_vline(
            data=bounds,
            mapping=aes(xintercept="t_start"),
            color="#999999",
            size=0.2,
            alpha=0.4,
            linetype="dashed",
        )
        + geom_vline(
            data=bounds,
            mapping=aes(xintercept="t_end"),
            color="#999999",
            size=0.2,
            alpha=0.4,
            linetype="dashed",
        )
        + facet_grid("week ~ weekday", scales="fixed")
        # Ensure 0 is always included in Y-axis
        + scale_x_continuous(limits=(0, max_t), expand=(0, 0))
        + scale_y_continuous(expand=(0.05, 0.1))
        + scale_color_manual(values=full_color_map)
        + labs(title=title, x="", y="Irradiance [W/m²]", color="")
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=7),
            axis_title=element_text(size=7),
            axis_text=element_text(size=7),
            axis_text_x=element_blank(),
            plot_title=element_text(size=8, face="bold"),
            strip_text_x=element_text(size=7, face="bold"),
            strip_text_y=element_blank(),
            figure_size=(width_inch, height_inch),
            panel_grid_major=element_line(size=0.15, alpha=0.7),
            legend_position="bottom",
            legend_text=element_text(size=7),
            panel_spacing=0,
            panel_border=element_blank(),
        )
    )

    # Ribbon for clear-sky ONLY if provided
    if clear_col:
        clear_label = label_map[clear_col]
        clear_data = long_df[long_df["kind_label"] == clear_label].copy()
        p = p + geom_ribbon(
            data=clear_data,
            mapping=aes(ymin=0, ymax="value", fill="kind_label"),
            color=None,
            alpha=0.15,
            show_legend=False,
        ) + scale_fill_manual(values=full_color_map)

    if output_file:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="plotnine")
            p.save(output_file, dpi=300, width=width_inch, height=height_inch, verbose=False)

    return p
