"""
Tests for BSRN Physics modules.
BSRN 物理模块测试。
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure 'src' is in path / 确保 'src' 在路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bsrn.physics.clearsky import add_clearsky_columns

# **Users can change these variables to their own directory / 用户可以根据自己的情况更改目录**
# Path to a sample file and directory for testing
# 用于测试的示例文件和目录路径
SAMPLE_FILE = "/Volumes/Macintosh Research/Data/bsrn-qc/data/QIQ/qiq0124.dat.gz"

class TestBSRNPhysics(unittest.TestCase):
    """
    Test suite for BSRN physics and clear-sky models.
    BSRN 物理和晴空模型测试套件。
    """

    def setUp(self):
        self.sample_file = SAMPLE_FILE

    def test_add_clearsky(self):
        """
        Test adding clear-sky columns to a QIQ data frame.
        测试向 QIQ 数据框添加晴空列。
        """
        if not os.path.exists(self.sample_file):
            self.skipTest(f"Sample file not found at {self.sample_file}")
            
        # Read data / 读取数据
        df = read_bsrn_station_to_archive(self.sample_file)
        self.assertIsNotNone(df)
        
        # Add clear-sky columns / 添加晴空列
        # QIQ is the station code / QIQ 是站点缩写
        df = add_clearsky_columns(df, station_code="QIQ")
        
        # Check columns / 检查列
        for col in ["ghi_clear", "bni_clear", "dhi_clear"]:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isnull().all(), f"Column {col} is all NaN")
            
        print("\nClear-sky results for QIQ (first 2 rows):")
        print(df[["ghi", "ghi_clear", "bni", "bni_clear"]].head(2))
        
        # Basic physical check: clear-sky GHI should be > 0 during day / 基础物理检查：白天晴空 GHI 应 > 0
        # Since this is Jan in QIQ (high latitude), we check a noon value / QIQ 1月（高纬度），检查中午值
        noon_val = df.between_time("04:00", "05:00")["ghi_clear"] # QIQ noon is approx 04:00 UTC
        self.assertTrue((noon_val > 0).any())

if __name__ == "__main__":
    unittest.main()
