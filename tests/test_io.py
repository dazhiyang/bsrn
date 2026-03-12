"""
Tests for BSRN IO modules.
BSRN IO 模块测试。
"""

import os
import sys
import unittest

# Ensure 'src' is in path / 确保 'src' 在路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bsrn.io.readers import read_bsrn_station_to_archive, read_bsrn_multiple_files

class TestBSRNReaders(unittest.TestCase):
    """
    Test suite for BSRN file readers.
    BSRN 文件读取器测试套件。
    """

    def setUp(self):
        # Path to a sample file and directory for testing
        # 用于测试的示例文件和目录路径
        self.data_dir = "/Volumes/Macintosh Research/Data/bsrn-qc/data/QIQ"
        self.sample_file = os.path.join(self.data_dir, "qiq0124.dat.gz")

    def test_read_lr0100(self):
        """
        Test reading LR0100 record from a station-to-archive file.
        测试从站点存档文件中读取 LR0100 记录。
        """
        if not os.path.exists(self.sample_file):
            self.skipTest(f"Sample file not found at {self.sample_file}")
            
        df = read_bsrn_station_to_archive(self.sample_file)
        
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Check if all required columns are present / 检查是否包含所有必需的列
        expected_cols = [
            "ghi", "bni", "dhi", "lwd", "temp", "rh", "pressure"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Verify DatetimeIndex and UTC timezone / 验证 DatetimeIndex 和 UTC 时区
        import pandas as pd
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.index.name, 'time')
        self.assertEqual(str(df.index.tz), 'UTC')
            
        print(f"\nRead {len(df)} lines from {os.path.basename(self.sample_file)}")
        print(df.head(2))

    def test_read_multiple_files(self):
        """
        Test reading and concatenating multiple BSRN files.
        测试读取并合并多个 BSRN 文件。
        """
        if not os.path.exists(self.data_dir):
            self.skipTest(f"Data directory not found at {self.data_dir}")
            
        # We expect qiq0124.dat.gz and qiq0224.dat.gz to be present
        # 我们期望 qiq0124.dat.gz 和 qiq0224.dat.gz 存在
        df = read_bsrn_multiple_files(self.data_dir, extension="qiq*.dat.gz")
        
        self.assertIsNotNone(df)
        
        expected_min_len = 44640 + 40000 
        self.assertGreater(len(df), expected_min_len)
        
        # Verify DatetimeIndex and UTC timezone / 验证 DatetimeIndex 和 UTC 时区
        import pandas as pd
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.index.name, 'time')
        self.assertEqual(str(df.index.tz), 'UTC')
        
        print(f"\nRead total {len(df)} lines from directory {self.data_dir}")
        print(df.tail(2))

if __name__ == "__main__":
    unittest.main()
