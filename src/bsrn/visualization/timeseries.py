import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_line, facet_wrap, theme, element_text, 
    theme_minimal, labs, scale_x_datetime, element_line,
    scale_color_manual
)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from bsrn.constants import BSRN_STATIONS
from bsrn.physics.geometry import get_solar_position, get_bni_extra
from bsrn.qc.ppl import ghi_ppl_test, bni_ppl_test, dhi_ppl_test


from bsrn.io.readers import read_bsrn_station_to_archive
from bsrn.physics.clearsky import add_clearsky_columns


def plot_bsrn_timeseries_booklet(file_path, output_file, station_code=None, apply_qc=False):
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

    Returns
    -------
    None
        The function saves the plots to the specified PDF file.
        该函数将图表保存到指定的 PDF 文件中。
    """
    # 0. Load Data / 加载数据
    plot_df = read_bsrn_station_to_archive(file_path)
    if plot_df is None:
        raise ValueError(f"Failed to read BSRN file: {file_path}")

    # Ensure it's exactly one month of data / 确保该文件仅包含一个月的数据
    unique_months = np.unique(plot_df.index.to_period('M'))
    if len(unique_months) != 1:
        raise ValueError(f"Input file must contain exactly one month of data. "
                         f"Found {len(unique_months)} months: {unique_months}")

    # 1. Geometry and Calculations / 几何与计算
    stn = BSRN_STATIONS.get(station_code, {'lat': 0, 'lon': 0, 'elev': 0})
    from bsrn.physics.geometry import get_solar_position
    solpos = get_solar_position(plot_df.index, stn['lat'], stn['lon'], stn['elev'])
    mu0 = np.cos(np.radians(solpos['apparent_zenith']))
    zenith = solpos['zenith']
    
    # Calculate GHI Sum and differences / 计算 GHI 总和及差异
    # Sum = BNI * cos(Z) + DHI
    plot_df['sum_irrad'] = plot_df['bni'] * mu0 + plot_df['dhi']
    plot_df['gh_ratio'] = plot_df['ghi'] / plot_df['sum_irrad']
    plot_df['gh_diff'] = plot_df['ghi'] - plot_df['sum_irrad']
    
    # 2. Add Clear-sky reference if needed / 如果需要，添加晴空参考
    if station_code is not None:
        plot_df = add_clearsky_columns(plot_df, station_code)

    # 3. QC Application / 应用质量控制
    if apply_qc:
        from bsrn.physics.geometry import get_bni_extra
        bni_extra = get_bni_extra(plot_df.index)
        
        ghi_mask = ghi_ppl_test(plot_df['ghi'], solpos['apparent_zenith'], bni_extra)
        bni_mask = bni_ppl_test(plot_df['bni'], bni_extra)
        dhi_mask = dhi_ppl_test(plot_df['dhi'], solpos['apparent_zenith'], bni_extra)
        
        plot_df.loc[~ghi_mask, 'ghi'] = np.nan
        plot_df.loc[~bni_mask, 'bni'] = np.nan
        plot_df.loc[~dhi_mask, 'dhi'] = np.nan

    # 4. Figure Setup / 图表设置
    width_inch = 160 / 25.4
    height_inch = width_inch * 1.0  # Taller for 3x3 grid / 3x3 网格需要更高
    
    plot_df = plot_df.sort_index()
    measured_color = '#E69F00' # Wong Orange
    clearsky_color = '#56B4E9' # Wong Sky Blue
    ribbon_color = clearsky_color
    
    # 5. Create PDF Booklet / 创建 PDF 手册
    print(f"Generating PDF booklet: {output_file}...")
    from plotnine import geom_ribbon, geom_line, geom_hline
    with PdfPages(output_file) as pdf:
        days = plot_df.groupby(plot_df.index.date)
        
        for date, day_df in days:
            formatted_date = date.strftime("%Y %b %d")
            day_zenith = zenith.loc[day_df.index]
            
            # --- Prepare Data for Main Plots (GHI, BNI, DHI, LWD) ---
            main_vars = ['ghi', 'bni', 'dhi']
            if 'lwd' in day_df.columns: main_vars.append('lwd')
            clear_vars = ['ghi_clear', 'bni_clear', 'dhi_clear']
            if 'lwd_clear' in day_df.columns: clear_vars.append('lwd_clear')
            
            day_main_measured = day_df.reset_index().melt(
                id_vars=['time'], 
                value_vars=main_vars, 
                var_name='parameter', value_name='measured'
            )
            day_main_clear = day_df.reset_index().melt(
                id_vars=['time'], 
                value_vars=clear_vars, 
                var_name='parameter', value_name='clearsky'
            )
            day_main_measured['parameter'] = day_main_measured['parameter'].str.upper()
            day_main_clear['parameter'] = day_main_clear['parameter'].str.replace('_clear', '').str.upper()
            day_main = pd.merge(day_main_measured, day_main_clear, on=['time', 'parameter'], how='left')
            
            # --- Prepare Data for Diagnostic & Meteorological Plots ---
            # Rearranged order: GHI-SUM, GHI/SUM, TEMP, RH, Pressure
            
            # Mask GHI/SUM ratio at night (zenith >= 90) / 夜间屏蔽 GHI/SUM 比率
            day_df_diag = day_df.copy()
            day_df_diag.loc[day_zenith >= 90, 'gh_ratio'] = np.nan
            
            diag_vars = ['gh_diff', 'gh_ratio', 'temp', 'rh', 'pressure']
            day_diag = day_df_diag.reset_index().melt(
                id_vars=['time'], 
                value_vars=diag_vars, 
                var_name='parameter', value_name='value'
            )
            # Map names / 映射名称
            name_map = {
                'gh_diff': 'GHI-SUM', 
                'gh_ratio': 'GHI/SUM', 
                'temp': 'TEMP',
                'rh': 'RH',
                'pressure': 'Pressure'
            }
            day_diag['parameter'] = day_diag['parameter'].map(name_map)
            
            # Order categories / 设置类别顺序
            all_params = ['GHI', 'BNI', 'DHI', 'GHI-SUM', 'GHI/SUM', 'LWD', 'TEMP', 'RH', 'Pressure']
            day_main['parameter'] = pd.Categorical(day_main['parameter'], categories=all_params, ordered=True)
            day_diag['parameter'] = pd.Categorical(day_diag['parameter'], categories=all_params, ordered=True)
            
            # Thresholds for GHI/SUM
            day_thresh = pd.DataFrame({
                'time': day_df.index,
                'upper': np.where(day_zenith < 75, 1.08, 1.15),
                'lower': np.where(day_zenith < 75, 0.92, 0.85),
                'parameter': pd.Categorical(['GHI/SUM']*len(day_df.index), categories=all_params, ordered=True)
            }, index=day_df.index)
            # Mask thresholds at night as well for clarity
            day_thresh.loc[day_zenith >= 90, ['upper', 'lower']] = np.nan

            title = f"{formatted_date}"
            if station_code: title = f"{station_code} - {title}"
            if apply_qc: title += " (QC Applied)"

            p = (
                ggplot(day_main, aes(x='time')) +
                # Row 1: Ribbons (Clear-sky)
                geom_ribbon(aes(ymin=0, ymax='clearsky'), fill=ribbon_color, alpha=0.25) +
                geom_line(aes(y='clearsky'), color=clearsky_color, size=0.3) +
                geom_line(aes(y='measured'), color=measured_color, size=0.3) +
                # Diagnostic & Meteo Plots
                geom_line(data=day_diag, mapping=aes(y='value'), color=measured_color, size=0.3) +
                # Ratio Envelopes (Row 2, Column 2)
                geom_line(data=day_thresh, mapping=aes(y='upper'), color='#999999', size=0.3, linetype='dashed') +
                geom_line(data=day_thresh, mapping=aes(y='lower'), color='#999999', size=0.3, linetype='dashed') +
                geom_hline(data=pd.DataFrame({'parameter': pd.Categorical(['GHI/SUM'], categories=all_params), 'y': [1.0]}), 
                           mapping=aes(yintercept='y'), color='#999999', size=0.2) +
                facet_wrap('~parameter', nrow=3, ncol=3, scales='free_y') +
                labs(title=title, x="Time (UTC)", y="Value") +
                theme_minimal() +
                theme(
                    text=element_text(family='Times New Roman', size=7),
                    axis_title=element_text(size=7),
                    axis_text=element_text(size=7),
                    plot_title=element_text(size=8, face='bold', margin={'b': 10}),
                    legend_position='none',
                    figure_size=(width_inch, height_inch),
                    panel_grid_minor=element_line(alpha=0),
                    strip_text=element_text(size=7, face='bold'),
                    axis_text_x=element_text(rotation=0)
                ) +
                scale_x_datetime(date_labels="%H:%M")
            )
            
            fig = p.draw()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print("Done.")
