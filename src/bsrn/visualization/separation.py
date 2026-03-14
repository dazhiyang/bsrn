"""
Visualization of irradiance separation results (Engerer2, etc.).
辐照分离结果的可视化（Engerer2 等）。
"""

import pandas as pd
import numpy as np
from bsrn.constants import WONG_PALETTE
from plotnine import (
    ggplot, aes, geom_line, geom_point,
    facet_wrap, theme_minimal, theme,
    element_text, element_line, element_blank,
    labs, scale_x_datetime, scale_color_manual,
    scale_x_continuous, scale_y_continuous,
    scale_color_cmap, coord_fixed
)

def plot_separation_results(
    df,
    ghi_col="ghi",
    dhi_col="dhi",
    bni_col="bni",
    k_col="k",
    title=None,
    output_file=None,
):
    """
    Plot irradiance separation results: GHI, DHI, BNI and diffuse fraction.
    绘制辐照分离结果：GHI、DHI、BNI 与散射分数。
    """
    required = [ghi_col, dhi_col, bni_col, k_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    plot_df = df[required].copy()
    plot_df = plot_df.reset_index().rename(columns={"index": "time"})
    
    # Melt for plotnine
    melt_df = plot_df.melt(id_vars=["time"], value_vars=[ghi_col, dhi_col, bni_col, k_col])
    
    # Mapping for display / 映射用于显示
    name_map = {ghi_col: "GHI", dhi_col: "DHI", bni_col: "BNI", k_col: "k"}
    melt_df["variable"] = melt_df["variable"].map(name_map)
    
    # Panel categorization / 分面板
    melt_df["panel"] = melt_df["variable"].apply(
        lambda x: "Irradiance (W/m²)" if x != "k" else "Diffuse Fraction k"
    )
    
    width_inch = 160 / 25.4
    
    p = (
        ggplot(melt_df, aes(x="time", y="value", color="variable"))
        + geom_line(size=0.4)
        + facet_wrap("panel", ncol=1, scales="free_y")
        + scale_color_manual(
            values={
                "GHI": WONG_PALETTE[0],
                "DHI": WONG_PALETTE[1],
                "BNI": WONG_PALETTE[2],
                "k": WONG_PALETTE[3],
            },
            name=""
        )
        + labs(title=title, x="Time", y="")
        + scale_x_datetime(date_labels="%H:%M")
        + theme_minimal()
        + theme(
            text=element_text(family="Times New Roman", size=7),
            plot_title=element_text(size=7),
            axis_title=element_text(size=7),
            axis_text=element_text(size=7),
            legend_title=element_text(size=7),
            legend_text=element_text(size=7),
            legend_position="bottom",
            figure_size=(width_inch, width_inch * 0.8)
        )
    )

    if output_file:
        p.save(output_file, dpi=300)
    return p

def plot_k_vs_kt(
    kt_meas,
    k_meas,
    kt_mod,
    k_mod,
    output_file=None,
    title=None,
):
    """
    Scatter plot of k vs kt using plotnine.
    使用 plotnine 绘制 k vs kt 散点图。
    """
    # Create dataframes / 创建数据框
    df_meas = pd.DataFrame({"kt": kt_meas, "k": k_meas}).dropna()
    df_mod = pd.DataFrame({"kt": kt_mod, "k": k_mod}).dropna()
    
    width_inch = 160 / 25.4
    
    p = (
        ggplot()
        # Bottom layer: measured in gray / 底层：灰色实测值
        + geom_point(
            data=df_meas, 
            mapping=aes(x="kt", y="k"), 
            color="gray", 
            alpha=0.2, 
            size=0.5,
            stroke=0,
            raster=True
        )
        # Top layer: modeled with viridis / 顶层：Viridis 颜色映射的模型值
        + geom_point(
            data=df_mod, 
            mapping=aes(x="kt", y="k", color="kt"), 
            alpha=0.5, 
            size=0.5,
            stroke=0,
            raster=True
        )
        + scale_color_cmap(cmap_name="viridis", name="$k_t$ (mod)")
        + labs(
            title=title, 
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
            legend_title=element_text(size=7),
            legend_text=element_text(size=7),
            panel_grid_minor=element_blank(),
            figure_size=(width_inch, width_inch * 0.8)
        )
    )

    if output_file:
        p.save(output_file, dpi=300)
    return p
