"""
Example: Compare Ground Measurements with CAMS CRS Satellite Data
示例：比较地面测量与 CAMS CRS 卫星数据

This script demonstrates reading BSRN station data, adding REST2 clear-sky 
references (from MERRA-2 HF), running the QC suite, adding CRS satellite 
data (from HF), calculating solar geometry, and performing time averaging.
本脚本演示读取 BSRN 站点数据、添加 REST2 晴空参考（来自 MERRA-2 HF）、运行 QC、
添加 CRS 卫星数据（来自 HF）、计算太阳几何以及执行时间平均。
"""

import os
import pandas as pd
import bsrn

# 1. Provide station code and input file path
# 提供站点代码与输入文件路径
station_code = "QIQ"
INPUT_FILE = "/Volumes/Macintosh Research/Data/bsrn-qc/data/QIQ/qiq0824.dat.gz"

print(f"--- Processing {station_code} for Aug 2024 ---")

# 2. Read station data (LR0100 record)
# 读取站点数据（LR0100 记录）
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

df = bsrn.io.readers.read_station_to_archive(INPUT_FILE, logical_records="lr0100")
print(f"Loaded {len(df)} rows from archive.")

# 3. Add solar positioning
# 计算太阳几何角
print("Calculating solar geometry...")
df = bsrn.physics.geometry.add_solpos_columns(df, station_code=station_code)

# 4. Add REST2 clear-sky columns (fetches MERRA-2 from Hugging Face)
# 添加 REST2 晴空列（从 Hugging Face 获取 MERRA-2）
print("Fetching MERRA-2 and adding REST2 clear-sky columns...")
df = bsrn.modeling.clear_sky.add_clearsky_columns(df, station_code=station_code, model="rest2")

# 5. Run QC suite
# 运行 QC 套件
print("Running QC suite...")
df = bsrn.qc.wrapper.run_qc(df, station_code=station_code)

print("Removing data that failed any QC test...")
flag_cols = [c for c in df.columns if c.startswith("flag")]
df = df[df[flag_cols].sum(axis=1) == 0].copy()
df.drop(columns=flag_cols, inplace=True)
print(f"Rows remaining after QC filtering: {len(df)}")

# 6. Perform time averaging (1-hour, ceiling alignment)
# 执行时间平均（1 小时，向上对齐）
print("Performing 1-hour time averaging...")
df_avg = bsrn.utils.pretty_average(df, rule="1h", alignment="ceiling")

# 7. Add CRS all-sky satellite columns (fetches CRS from Hugging Face)
# CRS data is hourly, so we append it to the averaged DataFrame.
# 添加 CRS 全天空卫星列（从 Hugging Face 获取 CRS）。CRS 为小时数据，故追加至平均后的 DataFrame。
print("Fetching and adding CRS all-sky columns...")
df_avg = bsrn.io.crs.add_crs_columns(df_avg, station_code=station_code)

print("\nProcessing complete. Averaged data with CRS (first 5 rows):")
print(df_avg.head())

# 8. Plot results in a calendar view
# 绘制日历视图结果
import bsrn.visualization.calendar

output_plot = "qiq_hf_comparison.pdf"
print(f"\nGenerating calendar plot: {output_plot}...")

bsrn.visualization.calendar.plot_calendar(
    df=df_avg,
    output_file=output_plot,
    station_code=station_code,
    meas_col='ghi',
    clear_col='ghi_clear',
    other_cols=['ghi_crs'],
    labels=['Measured GHI', 'REST2 Clear-sky GHI', 'CRS Satellite GHI'],
)

print(f"Plot saved to {output_plot}")
