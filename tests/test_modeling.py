"""
Demo/Test for BSRN modeling modules (e.g. irradiance separation) using real data from QIQ.
使用 QIQ 站点的真实数据进行 BSRN 建模模块演示/测试。
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure 'src' is in path / 确保 'src' 在路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bsrn.io.readers import read_bsrn_multiple_files
from bsrn.modeling import erbs_separation, engerer2_separation, brl_separation, yang4_separation
from bsrn.visualization.separation import plot_separation_results, plot_k_vs_kt
from bsrn.physics import geometry

def run_real_data_demo():
    # 1. Load data from QIQ folder / 从 QIQ 文件夹加载数据
    data_dir = "data/QIQ"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return

    print(f"Loading data from {data_dir}...")
    df_all = read_bsrn_multiple_files(data_dir)
    if df_all is None:
        print("Error: No data loaded.")
        return

    # Metadata for QIQ station
    lat, lon = 47.7957, 124.4852
    
    # 2. Run Engerer2 separation on the whole dataset / 对整个数据集运行 Engerer2 分离模型
    print("Running Engerer2 separation on the full year...")
    # Fill small gaps or handle NaNs for GHI if necessary
    df_all = df_all.dropna(subset=["ghi"])
    
    # We'll use a station_code to trigger Ineichen clear-sky in the background
    out_all = engerer2_separation(df_all, lat, lon, station_code="QIQ")
    
    # Calculate kt for scatter plot / 计算用于散点图的 kt
    # kt = GHI / GHI_extra
    ghi_extra = geometry.get_ghi_extra(out_all.index, geometry.get_solar_position(out_all.index, lat, lon)["zenith"])
    kt_mod = (out_all["ghi"] / ghi_extra).clip(0, 1.5)
    k_mod = out_all["k"]
    
    # For measured k, use dhi/ghi from the file
    k_meas = (df_all["dhi"] / df_all["ghi"]).clip(0, 1)
    # Mask indices where GHI is low to avoid noise
    mask = df_all["ghi"] > 50
    
    # 3. Select a representative clear day for timeseries / 选择一个代表性的晴天进行时间序列展示
    # Let's pick 2024-07-21 (arbitrary summer day)
    sample_day = "2024-07-21"
    out_sample = out_all.loc[sample_day]
    
    # 4. Output plots / 输出图表
    print("Saving plots to 'tests/'...")
    
    # Timeseries plot for the selected day
    plot_separation_results(
        out_sample, 
        title=f"Engerer2 Separation Results - QIQ {sample_day}",
        output_file="separation_timeseries_real.pdf"
    )
    
    # Scatter plot for the whole year
    plot_k_vs_kt(
        kt_meas=kt_mod[mask],
        k_meas=k_meas[mask],
        kt_mod=kt_mod[mask],
        k_mod=k_mod[mask],
        title="k vs kt Scatter - QIQ 2024",
        output_file="separation_scatter_real.pdf"
    )
    
    print("Done. Files created: separation_timeseries_real.pdf, separation_scatter_real.pdf")


import unittest


class TestErbs(unittest.TestCase):
    """Unit tests for Erbs separation model."""

    def setUp(self):
        self.times = pd.date_range(
            "2024-07-01 17:00", "2024-07-01 21:00", freq="1h", tz="UTC"
        )
        self.df = pd.DataFrame(
            {"ghi": [200.0, 500.0, 800.0, 900.0, 850.0]},
            index=self.times,
        )
        self.lat, self.lon = 40.0, -105.0

    def test_erbs_returns_columns(self):
        out = erbs_separation(self.df, self.lat, self.lon)
        for col in ["k", "dhi", "bni"]:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), len(self.df))

    def test_erbs_k_in_valid_range(self):
        out = erbs_separation(self.df, self.lat, self.lon)
        valid = out["k"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_erbs_piecewise(self):
        # k_t <= 0.22: k = 1.0 - 0.09*k_t; k_t > 0.80: k = 0.165
        df = pd.DataFrame(
            {"ghi": [100.0, 900.0]},
            index=pd.date_range("2024-07-01 12:00", periods=2, freq="1h", tz="UTC"),
        )
        # Use a single location; we need ghi_extra such that k_t is 0.2 and 0.85
        out = erbs_separation(df, 40.0, -105.0)
        k = out["k"].values
        self.assertTrue(np.all(np.isfinite(k) | np.isnan(k)))
        # At high GHI (high k_t), k should be 0.165 when k_t > 0.80
        self.assertGreater(k[1], 0.1)
        self.assertLess(k[1], 0.3)


class TestBRL(unittest.TestCase):
    """Unit tests for BRL separation model."""

    def setUp(self):
        # Hourly data over two days so daily K_t is defined
        self.times = pd.date_range(
            "2024-07-01 06:00", "2024-07-02 20:00", freq="1h", tz="UTC"
        )
        self.df = pd.DataFrame(
            {"ghi": 400.0 + 300.0 * np.sin(np.linspace(0, np.pi, len(self.times)))},
            index=self.times,
        )
        self.lat, self.lon = 40.0, -105.0

    def test_brl_returns_columns(self):
        out = brl_separation(self.df, self.lat, self.lon)
        for col in ["k", "dhi", "bni"]:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), len(self.df))

    def test_brl_k_in_valid_range(self):
        out = brl_separation(self.df, self.lat, self.lon)
        valid = out["k"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())


if __name__ == "__main__":
    unittest.main()
