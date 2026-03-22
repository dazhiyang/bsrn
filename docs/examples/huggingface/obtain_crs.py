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

# 4. Run QC suite and mask failed rows
# 运行 QC 套件并遮掩失败行
print("Running QC suite...")
df = bsrn.qc.wrapper.run_qc(df, station_code=station_code)

print("Masking data that failed any QC test as NaN...")
flag_cols = [c for c in df.columns if c.startswith("flag")]
bad_mask = df[flag_cols].sum(axis=1) > 0
irradiance_cols = ["ghi", "bni", "dhi"]
irradiance_cols = [c for c in irradiance_cols if c in df.columns]
df.loc[bad_mask, irradiance_cols] = pd.NA
print(f"Masked {bad_mask.sum()} rows as NaN.")

# Drop flag columns after they've been used for masking
# 使用完成后移除 QC 标志列
df.drop(columns=flag_cols, inplace=True)

# 5. Add REST2 clear-sky columns (fetches MERRA-2 from Hugging Face)
# 添加 REST2 晴空列（从 Hugging Face 获取 MERRA-2）
print("Fetching MERRA-2 and adding REST2 clear-sky columns...")
df = bsrn.modeling.clear_sky.add_clearsky_columns(df, station_code=station_code, model="rest2")

# 6. Add CRS all-sky satellite columns (fetches CRS from Hugging Face)
# CRS data is now 1-minute resolution, so we add it before averaging for precision.
# 添加 CRS 全天空卫星列（从 Hugging Face 获取）。CRS 已提升至 1 分钟分辨率，建议在平均前加入。
print("Fetching and adding CRS all-sky columns (1-min)...")
df = bsrn.io.crs.add_crs_columns(df, station_code=station_code)

# 7. Perform time averaging (e.g., 30-min centered average)
# 执行时间平均（例如，30 分钟居中平均）
print("Performing 30-min centered time averaging...")
df_avg = bsrn.utils.pretty_average(
    df, 
    rule="30min", 
    alignment="center",
    match_ceiling_labels=True
)

print("\nProcessing complete. sample averaged data:")
print(df_avg[['ghi', 'ghi_clear', 'ghi_crs']].head())

# 8. Plot results in a calendar view
# 绘制日历视图结果
import bsrn.visualization.calendar

output_plot = "qiq_hf_comparison.pdf"
print(f"\nGenerating calendar plot: {output_plot}...")

# Note: plot_calendar expects floor-aligned data (start-of-interval)
# Note: plot_calendar 期望 start-of-interval 对齐的数据
bsrn.visualization.calendar.plot_calendar(
    df=df_avg,
    output_file=output_plot,
    station_code=station_code,
    meas_col='ghi',
    clear_col='ghi_clear',
    other_cols=['ghi_crs'],
    labels=['Measured GHI', 'REST2 Clear-sky GHI', 'CAMS CRS Satellite GHI'],
)

print(f"Plot saved to {output_plot}")
