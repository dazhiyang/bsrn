"""
Quality control summary and LaTeX table generation.
质量控制摘要和 LaTeX 表格生成。
"""

import os
import pandas as pd
import numpy as np
import bsrn.io.readers as readers
import bsrn.physics.geometry as geometry
import bsrn.physics.clearsky as clearsky
import bsrn.qc.ppl as ppl
import bsrn.qc.erl as erl
import bsrn.qc.closure as closure
import bsrn.qc.k_index as k_index
import bsrn.qc.tracker as tracker
from bsrn.constants import BSRN_STATIONS
from glob import glob
from datetime import datetime

def get_quality_stats(file_path, station_code):
    """
    Run QC tests on a single BSRN file and count flags.
    对单个 BSRN 文件运行 QC 测试并统计标记数量。

    Parameters
    ----------
    file_path : str
        Path to the BSRN file (.dat.gz).
        BSRN 文件的路径 (.dat.gz)。
    station_code : str
        BSRN station abbreviation (e.g., 'QIQ').
        BSRN 站点缩写（例如 'QIQ'）。

    Returns
    -------
    stats_df : pd.DataFrame
        Summary of QC flag counts and missing values.
        QC 标记计数和缺失值摘要。
    """
    if station_code not in BSRN_STATIONS:
        raise ValueError(f"Station {station_code} not found in BSRN_STATIONS. / 站点 {station_code} 未在 BSRN_STATIONS 中找到。")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path} / 未找到文件")
        return None

    filename = os.path.basename(file_path)
    # Read file / 读取文件
    # The reader handles -999 and -99.9 by replacing them with NaN
    # 读取器通过将 -999 和 -99.9 替换为 NaN 来处理缺失值
    df = readers.read_bsrn_station_to_archive(file_path)
    if df is None:
        return None

        total_records = len(df)
        
        # Count records with missing values (NaNs) / 统计含有缺失值 (NaN) 的记录数
        # The reader already replaced -999 and -99.9 with NaN
        # 读取器已将 -999 和 -99.9 置换成 NaN
        # We count unique timestamps where any measured value is missing
        # 我们统计任何观测值缺失的唯一时间戳数量
        total_missing = df.isnull().any(axis=1).sum()

        # Calculate solar geometry / 计算太阳几何参数
        solpos = geometry.get_solar_position(df.index, lat, lon, elev)
        zenith = solpos["zenith"]
        bni_extra = geometry.get_bni_extra(df.index)

        # Run QC tests / 运行 QC 测试
        # Note: Tests return True for "passed". We count flags where test is False (failed).
        # We count unique timestamps where any test fails.
        # 注意：测试对“通过”返回 True。我们统计测试为 False（失败）的标记。
        # 我们统计任何测试失败的唯一时间戳数量。
        
        # L1 (Physically Possible) / 1 级（物理可能）
        l1_any_flag = (
            (~ppl.ghi_ppl_test(df["ghi"], zenith, bni_extra)) |
            (~ppl.bni_ppl_test(df["bni"], bni_extra)) |
            (~ppl.dhi_ppl_test(df["dhi"], zenith, bni_extra)) |
            (~ppl.lwd_ppl_test(df["lwd"]))
        )
        l1_total = l1_any_flag.sum()

        # L2 (Extremely Rare) / 2 级（极罕见）
        l2_any_flag = (
            (~erl.ghi_erl_test(df["ghi"], zenith, bni_extra)) |
            (~erl.bni_erl_test(df["bni"], zenith, bni_extra)) |
            (~erl.dhi_erl_test(df["dhi"], zenith, bni_extra)) |
            (~erl.lwd_erl_test(df["lwd"]))
        )
        l2_total = l2_any_flag.sum()

        # L3 (Comparison) / 3 级（比较）
        l3_any_flag = (
            (~closure.closure_low_sza_test(df["ghi"], df["bni"], df["dhi"], zenith)) |
            (~closure.closure_high_sza_test(df["ghi"], df["bni"], df["dhi"], zenith)) |
            (~k_index.kb_kt_test(df["ghi"], df["bni"], bni_extra, zenith)) |
            (~k_index.k_kt_combined_test(df["ghi"], df["dhi"], bni_extra, zenith))
        )
        l3_total = l3_any_flag.sum()

    results = [{
        "File": filename,
        "Total": total_records,
        "Missing": total_missing,
        "L1_Flags": l1_total,
        "L2_Flags": l2_total,
        "L3_Flags": l3_total
    }]

    return pd.DataFrame(results)

def get_daily_stats(df, lat, lon, elev):
    """
    Calculate daily QC statistics and sunshine duration.
    计算每日 QC 统计信息和日照时数。

    Returns
    -------
    daily_df : pd.DataFrame
        Daily counts of flags and sunshine duration metrics.
    """
    # Calculate solar geometry / 计算太阳几何参数
    solpos = geometry.get_solar_position(df.index, lat, lon, elev)
    zenith = solpos["zenith"]
    bni_extra = geometry.get_bni_extra(df.index)

    # Sunshine Duration (h) / 日照时数 (h)
    # ACT: BNI > 120 W/m² / 实际：BNI > 120 W/m²
    # MAX: Zenith < 90 / 最大：天顶角 < 90
    df['is_sunshine'] = (df['bni'] > 120).astype(int)
    df['is_daylight'] = (zenith < 90).astype(int)
    
    # QC Tests (True = Pass, False = Fail) / QC 测试（True = 通过，False = 失败）
    # We create a dictionary of daily failure counts / 创建每日失败计数的字典
    tests = {
        'SWD_PPL': ~ppl.ghi_ppl_test(df["ghi"], zenith, bni_extra),
        'SWD_ERL': ~erl.ghi_erl_test(df["ghi"], zenith, bni_extra),
        'DIF_PPL': ~ppl.dhi_ppl_test(df["dhi"], zenith, bni_extra),
        'DIF_ERL': ~erl.dhi_erl_test(df["dhi"], zenith, bni_extra),
        'DIR_PPL': ~ppl.bni_ppl_test(df["bni"], bni_extra),
        'DIR_ERL': ~erl.bni_erl_test(df["bni"], zenith, bni_extra),
        'LWD_PPL': ~ppl.lwd_ppl_test(df["lwd"]),
        'LWD_ERL': ~erl.lwd_erl_test(df["lwd"]),
        'CMP_CLO_L': ~closure.closure_low_sza_test(df["ghi"], df["bni"], df["dhi"], zenith),
        'CMP_CLO_H': ~closure.closure_high_sza_test(df["ghi"], df["bni"], df["dhi"], zenith),
        'CMP_KB_KT': ~k_index.kb_kt_test(df["ghi"], df["bni"], bni_extra, zenith),
        'CMP_COMB': ~k_index.k_kt_combined_test(df["ghi"], df["dhi"], bni_extra, zenith),
        'TRACKER': ~tracker.tracker_off_test(df["ghi"], df["bni"], zenith, ghi_extra=bni_extra)
    }
    
    for name, flags in tests.items():
        df[name] = flags.astype(int)

    # Aggregate by day / 按天汇总
    daily = df.groupby(df.index.date).agg({
        'is_sunshine': 'sum',
        'is_daylight': 'sum',
        'SWD_PPL': 'sum', 'SWD_ERL': 'sum',
        'DIF_PPL': 'sum', 'DIF_ERL': 'sum',
        'DIR_PPL': 'sum', 'DIR_ERL': 'sum',
        'LWD_PPL': 'sum', 'LWD_ERL': 'sum',
        'CMP_CLO_L': 'sum', 'CMP_CLO_H': 'sum',
        'CMP_KB_KT': 'sum', 'CMP_COMB': 'sum',
        'TRACKER': 'sum'
    })
    
    daily['SD_ACT'] = daily['is_sunshine'] / 60.0
    daily['SD_MAX'] = daily['is_daylight'] / 60.0
    daily['SD_REL'] = (daily['SD_ACT'] / daily['SD_MAX'] * 100.0).fillna(0)
    
    return daily

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate BSRN QC summary. / 计算 BSRN QC 摘要。")
    parser.add_argument("file_path", help="Path to a BSRN file (.dat.gz). / BSRN 文件路径。")
    parser.add_argument("station_code", help="BSRN station abbreviation (e.g., 'QIQ'). / BSRN 站点缩写。")
    parser.add_argument("--daily", action="store_true", help="Calculate detailed daily stats. / 计算详细的每日统计信息。")
    
    args = parser.parse_args()
    
    try:
        if args.daily:
            print(f"Calculating daily stats for: {args.file_path}")
            df = readers.read_bsrn_station_to_archive(args.file_path)
            if df is not None:
                meta = BSRN_STATIONS[args.station_code]
                daily_stats = get_daily_stats(df, meta["lat"], meta["lon"], meta["elev"])
                print("\nDaily Statistics Summary / 每日统计摘要:")
                print(daily_stats)
        else:
            stats = get_quality_stats(args.file_path, args.station_code)
            if stats is not None:
                print("\nMonthly Statistics Summary / 每月统计摘要:")
                print(stats)
    except Exception as e:
        print(f"Error: {e} / 错误：{e}")
        import traceback
        traceback.print_exc()
