"""
Quality control summary for BSRN data.
BSRN 数据质量控制摘要。
"""

import pandas as pd
import bsrn.physics.geometry as geometry
import bsrn.qc.ppl as ppl
import bsrn.qc.erl as erl
import bsrn.qc.closure as closure
import bsrn.qc.k_index as k_index
import bsrn.qc.tracker as tracker


def get_daily_stats(df, lat, lon, elev):
    """
    Calculate daily QC statistics and sunshine duration.
    计算每日 QC 统计信息和日照时数。

    Parameters
    ----------
    df : pd.DataFrame
        BSRN data archive.
        BSRN 数据存档。
    lat : float
        Latitude. [degrees]
        纬度。[度]
    lon : float
        Longitude. [degrees]
        经度。[度]
    elev : float
        Elevation. [m]
        海拔。[米]

    Returns
    -------
    daily_df : pd.DataFrame
        Daily counts of flags and sunshine duration metrics.
        每日标记计数和日照时数指标。
    """
    # Calculate solar geometry / 计算太阳几何参数
    solpos = geometry.get_solar_position(df.index, lat, lon, elev)
    zenith = solpos["zenith"]
    bni_extra = geometry.get_bni_extra(df.index)
    ghi_extra = geometry.get_ghi_extra(df.index, zenith)

    # Sunshine duration (h) / 日照时数 (h)
    # ACT: BNI > 120 W/m^2 / 实际：BNI > 120 W/m^2
    # MAX: Zenith < 90 / 最大：太阳天顶角 < 90
    df['is_sunshine'] = (df['bni'] > 120).astype(int)
    df['is_daylight'] = (zenith < 90).astype(int)
    
    # QC Tests (True = Pass, False = Fail) / QC 测试（True = 通过，False = 失败）
    # We create a dictionary of daily failure counts / 创建每日失败计数的字典
    tests = {
        'GHI_PPL': ~ppl.ghi_ppl_test(df["ghi"], zenith, bni_extra),
        'GHI_ERL': ~erl.ghi_erl_test(df["ghi"], zenith, bni_extra),
        'DHI_PPL': ~ppl.dhi_ppl_test(df["dhi"], zenith, bni_extra),
        'DHI_ERL': ~erl.dhi_erl_test(df["dhi"], zenith, bni_extra),
        'BNI_PPL': ~ppl.bni_ppl_test(df["bni"], bni_extra),
        'BNI_ERL': ~erl.bni_erl_test(df["bni"], zenith, bni_extra),
        'LWD_PPL': ~ppl.lwd_ppl_test(df["lwd"]),
        'LWD_ERL': ~erl.lwd_erl_test(df["lwd"]),
        # Combined L3 groups / 合并的 3 级分组
        'CMP_CLO': ((~closure.closure_low_sza_test(df["ghi"], df["bni"], df["dhi"], zenith)) |
                    (~closure.closure_high_sza_test(df["ghi"], df["bni"], df["dhi"], zenith))),
        'CMP_DIF': ((~k_index.k_low_sza_test(df["ghi"], df["dhi"], zenith)) |
                    (~k_index.k_high_sza_test(df["ghi"], df["dhi"], zenith))),
        'CMP_K': ((~k_index.kb_kt_test(df["ghi"], df["bni"], bni_extra, zenith)) |
                  (~k_index.k_kt_combined_test(df["ghi"], df["dhi"], bni_extra, zenith)) |
                  (~k_index.kb_limit_test(df["bni"], bni_extra, elev, df["ghi"])) |
                  (~k_index.kt_limit_test(df["ghi"], bni_extra, zenith))),
        'TRACKER': ~tracker.tracker_off_test(df["ghi"], df["bni"], zenith, ghi_extra=ghi_extra)
    }
    
    for name, flags in tests.items():
        df[name] = flags.astype(int)

    # Aggregate by day / 按天汇总
    daily = df.groupby(df.index.date).agg({
        'is_sunshine': 'sum',
        'is_daylight': 'sum',
        'GHI_PPL': 'sum', 'GHI_ERL': 'sum',
        'DHI_PPL': 'sum', 'DHI_ERL': 'sum',
        'BNI_PPL': 'sum', 'BNI_ERL': 'sum',
        'LWD_PPL': 'sum', 'LWD_ERL': 'sum',
        'CMP_CLO': 'sum', 'CMP_DIF': 'sum', 'CMP_K': 'sum',
        'TRACKER': 'sum'
    })
    
    daily['SD_ACT'] = daily['is_sunshine'] / 60.0
    daily['SD_MAX'] = daily['is_daylight'] / 60.0
    daily['SD_REL'] = (daily['SD_ACT'] / daily['SD_MAX'] * 100.0).fillna(0)
    
    return daily
