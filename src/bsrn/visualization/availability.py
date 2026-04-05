"""
visualization of bsrn file availability.
BSRN文件可用性的可视化。
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from plotnine import (
    ggplot, aes, geom_tile, scale_fill_cmap,
    theme, element_text, element_blank, element_line,
    labs, scale_x_continuous, scale_y_discrete,
    coord_equal, theme_minimal
)
from bsrn.io.retrieval import get_bsrn_file_inventory


def plot_bsrn_availability(
    stations, username, password, start_year=1992, end_year=None, output_file=None, title=None
):
    """
    Unified function to plot bsrn file availability from ftp.
    统一功能，用于绘制FTP中的BSRN文件可用性。

    Parameters
    ----------
    stations : str or list
        One or more station abbreviations (e.g., 'PAY' or ['PAY', 'NYA']).
        一个或多个站点缩写（例如 'PAY' 或 ['PAY', 'NYA']）。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。
    start_year : int, default 1992
        year to start the visualization.
        可视化开始的年份。
    end_year : int, optional
        year to end the visualization. If not specified, the current year is used.
        可视化结束的年份。如果未指定，则使用当前年份。
    output_file : str, optional
        Path to the output file (e.g., 'availability.pdf').
        输出文件的路径（例如 'availability.pdf'）。
    title : str, optional
        Plot title. If None (default), no title is drawn.
        图标题；默认 None 不显示。

    Returns
    -------
    fig : plotnine.ggplot.ggplot
        The generated availability heatmap figure.
        生成的可用性热图对象。
    """
    if isinstance(stations, str):
        stations = [stations]
    
    stations = [s.upper() for s in stations]
    stations.sort()  # Sort stations alphabetically / 按字母顺序对站点进行排序
    
    if end_year is None:
        end_year = datetime.now().year
        
    years = list(range(start_year, end_year + 1))
    
    # Fetch data from FTP / 从 FTP 获取数据
    print(f"Searching BSRN FTP for stations: {', '.join(stations)}...")
    inventory = get_bsrn_file_inventory(stations, username, password)
    if not inventory or all(len(files) == 0 for files in inventory.values()):
        warnings.warn(
            "BSRN FTP connection failed or returned no data. Plot may be empty. "
            "Check credentials (username/password) and network. / "
            "BSRN FTP 未连接或未返回数据，图可能为空。请检查凭据与网络。",
            UserWarning,
            stacklevel=2,
        )

    # Pattern: STNMMYY.dat.gz or STNMMYY.001 / 匹配模式：STNMMYY.dat.gz 或 STNMMYY.001
    pattern = re.compile(r"([A-Z]{3})(\d{2})(\d{2})\.(?:dat\.gz|\d{3}).*", re.IGNORECASE)

    # Calculate dimensions to ensure square cells / 计算尺寸以确保方形单元格
    # Width = 160mm = 6.299 inches / 宽度 = 160mm = 6.299 英寸
    width_inch = 160 / 25.4
    num_cols = len(years)
    num_rows = 12 if len(stations) == 1 else len(stations)
    
    # Estimate usable width for the plot (subtracting space for Y-axis labels)
    usable_width = width_inch - 0.8 
    cell_size = usable_width / num_cols
    
    # Total height scales with number of rows + overhead for title and legend
    total_height_inch = (cell_size * num_rows) + 1.0 
    
    # Data collection
    plot_data = []
    
    if len(stations) == 1:
        stn = stations[0]
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        df = pd.DataFrame(0, index=range(1, 13), columns=years)
        for filename in inventory.get(stn, []):
            basename = os.path.basename(filename)
            match = pattern.match(basename)
            if match:
                stn_match, mm_str, yy_str = match.groups()
                month, yy = int(mm_str), int(yy_str)
                # BSRN started 1991: yy>=91 -> 1900s / BSRN 自 1991 年起
                year = (1900 + yy) if yy >= 91 else (2000 + yy)
                if year in df.columns and month in df.index:
                    df.at[month, year] = 1
                    
        # Reset and melt for plotnine
        df = df.reset_index().rename(columns={'index': 'Month'})
        df_melt = df.melt(id_vars=['Month'], var_name='Year', value_name='Availability')
        
        # Use string representations of month integers to keep uniqueness
        month_strs = [str(m) for m in range(1, 13)]
        df_melt['Month'] = pd.Categorical(df_melt['Month'].astype(str), categories=reversed(month_strs), ordered=True)
        df_melt['Year'] = df_melt['Year'].astype(int)
        
        # Create mapping dictionary for ggplot axis labels
        month_labels = {str(m): name for m, name in zip(range(1, 13), month_names)}
        
        _labs = {"y": f"Station: {stn}", "x": "Year"}
        if title is not None:
            _labs["title"] = title
        p = (
            ggplot(df_melt, aes(x='Year', y='Month', fill='Availability')) + 
            geom_tile(color='white', size=0.5) +
            scale_fill_cmap(cmap_name='viridis', name="Availability") +
            labs(**_labs) +
            scale_y_discrete(labels=lambda items: [month_labels.get(x, x) for x in items])
        )
    else:
        df = pd.DataFrame(0, index=stations, columns=years)
        for stn in stations:
            for filename in inventory.get(stn, []):
                basename = os.path.basename(filename)
                match = pattern.match(basename)
                if match:
                    stn_match, mm_str, yy_str = match.groups()
                    yy = int(yy_str)
                    # BSRN started 1991: yy>=91 -> 1900s / BSRN 自 1991 年起
                    year = (1900 + yy) if yy >= 91 else (2000 + yy)
                    if year in df.columns:
                        df.at[stn, year] += 1
                        
        df = df.reset_index().rename(columns={'index': 'Station'})
        df_melt = df.melt(id_vars=['Station'], var_name='Year', value_name='Months_Available')
        df_melt['Station'] = pd.Categorical(df_melt['Station'], categories=reversed(stations), ordered=True)
        df_melt['Year'] = df_melt['Year'].astype(int)
        
        _labs = {"y": "Stations", "x": "Year"}
        if title is not None:
            _labs["title"] = title
        p = (
            ggplot(df_melt, aes(x='Year', y='Station', fill='Months_Available')) + 
            geom_tile(color='white', size=0.5) +
            scale_fill_cmap(cmap_name='viridis', name="Months Available") +
            labs(**_labs) +
            scale_y_discrete()
        )
        
    year_breaks = [y for i, y in enumerate(years) if i % 2 == 0] if len(years) > 20 else years
    
    p = p + scale_x_continuous(breaks=year_breaks) + coord_equal()

    # Apply strict formatting rules / 应用严格的格式规则
    p = p + theme_minimal() + theme(
        text=element_text(family='Times New Roman', size=9),
        axis_title=element_text(size=9),
        axis_text=element_text(size=9),
        plot_title=element_text(size=9, margin={'b': 1}),
        legend_title=element_text(size=9),
        legend_text=element_text(size=9),
        legend_position="bottom",
        legend_key_width=100,
        legend_key_height=5,
        legend_margin=-12,
        legend_box_spacing=0,
        axis_title_x=element_text(size=9, margin={"t": 1, "b": 2}),  # space between x-title and legend
        plot_margin_top=0,
        plot_margin_right=0,
        plot_margin_bottom=0,
        plot_margin_left=0,
        panel_grid=element_line(color="white", size=0.5),
        figure_size=(width_inch, total_height_inch),
        axis_text_x=element_text(rotation=45, hjust=1)
    )

    if output_file:
        p.save(output_file, dpi=300)
        
    return p
