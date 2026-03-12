"""
Tests for BSRN IO modules.
BSRN IO 模块测试。
"""

import os
import sys
import unittest

# Ensure 'src' is in path / 确保 'src' 在路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bsrn.io.readers import read_bsrn_station_to_archive

class TestBSRNReaders(unittest.TestCase):
    """
    Test suite for BSRN file readers.
    BSRN 文件读取器测试套件。
    """

    def setUp(self):
        # Path to a sample file for testing / 用于测试的示例文件路径
        self.sample_file = "/Volumes/Macintosh Research/Data/bsrn-qc/data/QIQ/qiq0124.dat.gz"

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
            "day_number", "minute_number", "ghi", "bni", "dhi", 
            "lwd", "temp", "rh", "pressure"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        print(f"\nRead {len(df)} lines from {os.path.basename(self.sample_file)}")
        print(df.head(2))

if __name__ == "__main__":
    unittest.main()
