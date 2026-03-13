"""
quality control table visualization.
质量控制表格可视化。
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from plotnine import (
    ggplot, aes, geom_tile, geom_text,
    theme, element_text, element_blank,
    labs, scale_x_discrete, scale_y_discrete,
    scale_fill_manual, theme_minimal, coord_fixed
)

# Project defined Wong palette / 项目定义的 Wong 配色方案
WONG_PALETTE = [
    "#E69F00",  # Orange / 橙色
    "#56B4E9",  # Sky Blue / 天蓝色
    "#009E73",  # Bluish Green / 青绿色
    "#CC79A7",  # Reddish Purple / 浅紫红色
    "#D55E00",  # Vermillion / 朱红色
    "#F0E442",  # Yellow / 黄色
    "#0072B2"   # Blue / 蓝色
]

def plot_qc_table(daily_stats, title="BSRN Quality Audit", output_file=None):
    """
    Plot QC statistics in a table-like heatmap format.
    以类似于表格的热图格式绘制 QC 统计数据。

    Parameters
    ----------
    daily_stats : pd.DataFrame
        Daily QC statistics calculated by `get_daily_stats`.
        由 `get_daily_stats` 计算的每日 QC 统计数据。
    title : str, default "BSRN Quality Audit"
        Plot title.
        图表标题。
    output_file : str, optional
        Path to save the plot (e.g., 'qc_table.png').
        保存图表的路径（例如 'qc_table.png'）。

    Returns
    -------
    p : ggplot
        The generated quality audit figure.
        生成的质量审计图。
    """
    df = daily_stats.copy()
    
    # Format date as day number for the Y axis / 将日期格式化为 Y 轴的天数
    df['Day'] = [str(d.day) for d in df.index]
    
    # Standard names mapping / 标准名称映射
    cols_to_plot = {
        'SD_MAX': 'SD MAX',
        'SD_ACT': 'SD ACT',
        'SD_REL': 'SD REL',
        'SWD_PPL': 'SWD PPL',
        'SWD_ERL': 'SWD ERL',
        'DIF_PPL': 'DIF PPL',
        'DIF_ERL': 'DIF ERL',
        'DIR_PPL': 'DIR PPL',
        'DIR_ERL': 'DIR ERL',
        'LWD_PPL': 'LWD PPL',
        'LWD_ERL': 'LWD ERL',
        'CMP_CLO_L': 'CLO L',
        'CMP_CLO_H': 'CLO H',
        'CMP_KB_KT': 'KB KT',
        'CMP_COMB': 'COMB',
        'TRACKER': 'TRACKER'
    }
    
    available_cols = [c for c in cols_to_plot.keys() if c in df.columns]
    plot_df = df[['Day'] + available_cols].copy()
    
    # Melt for plotnine / 为 plotnine 进行转换
    melted = plot_df.melt(id_vars=['Day'], var_name='Metric', value_name='Value')
    melted['Metric_Name'] = melted['Metric'].map(cols_to_plot)
    
    # Preservation of order / 顺序保留
    order = [cols_to_plot[c] for c in available_cols]
    melted['Metric_Name'] = pd.Categorical(melted['Metric_Name'], categories=order, ordered=True)
    
    # Top-to-bottom days / 从上到下的天数
    days_order = sorted([int(d) for d in melted['Day'].unique()], reverse=True)
    melted['Day'] = pd.Categorical(melted['Day'], categories=[str(d) for d in days_order], ordered=True)
    
    # Define categories based on naming and pass/fail / 根据命名和通过/失败定义类别
    def get_category(row):
        val = row['Value']
        met = row['Metric']
        if val == 0:
            return 'Pass'
        if 'SD' in str(met):
            return 'Stats'
        if 'PPL' in str(met):
            return 'L1_Fail'
        if 'ERL' in str(met):
            return 'L2_Fail'
        return 'L3_Fail'

    melted['Category'] = melted.apply(get_category, axis=1)
    
    # Apply Wong Palette / 应用 Wong 配色方案
    # White for Pass, others mapped logically
    # Pass: white, Stats: WONG[1] (Sky Blue), L1: WONG[4] (Vermillion), 
    # L2: WONG[0] (Orange), L3: WONG[5] (Yellow)
    fill_colors = {
        'Pass': '#FFFFFF',
        'Stats': WONG_PALETTE[1],
        'L1_Fail': WONG_PALETTE[4],
        'L2_Fail': WONG_PALETTE[0],
        'L3_Fail': WONG_PALETTE[5]
    }

    # Format labels / 格式化标签
    def format_val(row):
        val = row['Value']
        met = row['Metric']
        if 'REL' in str(met):
            return f"{val:.0f}%"
        if 'SD' in str(met):
            return f"{val:.1f}"
        return f"{val:.0f}"

    melted['Label'] = melted.apply(format_val, axis=1)

    # Figure dimensions: width 160mm / 图片尺寸：宽度 160mm
    # Height varies with number of days / 高度随天数变化
    width_inch = 160 / 25.4
    height_inch = (len(days_order) * 0.18) + 1.2

    p = (
        ggplot(melted, aes(x='Metric_Name', y='Day', fill='Category')) +
        geom_tile(color='#D0D0D0', size=0.3) +
        geom_text(aes(label='Label'), size=7, family='Times New Roman') +
        scale_fill_manual(values=fill_colors, guide=None) +
        scale_x_discrete() +
        labs(title=title, x="", y="Date") +
        theme_minimal() +
        theme(
            text=element_text(family='Times New Roman', size=7),
            axis_text_x=element_text(rotation=45, hjust=0),
            plot_title=element_text(size=7, margin={'b': 5}),
            panel_grid=element_blank(),
            figure_size=(width_inch, height_inch)
        )
    )

    if output_file:
        p.save(output_file, dpi=300)
    
    return p
