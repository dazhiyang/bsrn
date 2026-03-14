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
from bsrn.modeling import erbs_separation, brl_separation
from bsrn.physics import geometry

def run_real_data_demo():
    """Load QIQ data and run Erbs and BRL separation (no plotting)."""
    data_dir = "data/QIQ"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return

    print(f"Loading data from {data_dir}...")
    df_all = read_bsrn_multiple_files(data_dir)
    if df_all is None:
        print("Error: No data loaded.")
        return

    lat, lon = 47.7957, 124.4852
    df_all = df_all.dropna(subset=["ghi"])

    print("Running Erbs and BRL separation...")
    result_erbs = erbs_separation(
        df_all.index, df_all["ghi"].values, lat, lon
    )
    result_brl = brl_separation(
        df_all.index, df_all["ghi"].values, lat, lon
    )
    print("Done. Use visualization.separation.plot_k_vs_kt to plot (e.g. in test_visualization).")


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
        out = erbs_separation(
            self.df.index, self.df["ghi"], self.lat, self.lon
        )
        for col in ["k", "dhi", "bni"]:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), len(self.df))

    def test_erbs_k_in_valid_range(self):
        out = erbs_separation(
            self.df.index, self.df["ghi"], self.lat, self.lon
        )
        valid = out["k"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_erbs_piecewise(self):
        # kt <= 0.22: k = 1.0 - 0.09*kt; kt > 0.80: k = 0.165
        df = pd.DataFrame(
            {"ghi": [100.0, 900.0]},
            index=pd.date_range("2024-07-01 12:00", periods=2, freq="1h", tz="UTC"),
        )
        # Use a single location; we need ghi_extra such that kt is 0.2 and 0.85
        out = erbs_separation(df.index, df["ghi"], 40.0, -105.0)
        k = out["k"].values
        self.assertTrue(np.all(np.isfinite(k) | np.isnan(k)))
        # At high GHI (high kt), k should be 0.165 when kt > 0.80
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
        out = brl_separation(
            self.df.index, self.df["ghi"], self.lat, self.lon
        )
        for col in ["k", "dhi", "bni"]:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), len(self.df))

    def test_brl_k_in_valid_range(self):
        out = brl_separation(
            self.df.index, self.df["ghi"], self.lat, self.lon
        )
        valid = out["k"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())


if __name__ == "__main__":
    unittest.main()
