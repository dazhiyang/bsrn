import os
import sys

# Ensure the 'src' directory is in the path so we can import 'bsrn'
# 确保 'src' 目录在路径中，以便我们可以导入 'bsrn'
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from bsrn.io.retrieval import download_bsrn_mon
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure all dependencies (pandas, etc.) are installed and you are running from the project root.")
    sys.exit(1)

# Users can change these variables to their own directory / 用户可以根据自己的情况更改目录
# --- CONFIGURATION ---
BSRN_USER = "your_username" # Replace with your BSRN FTP username / 替换为您的 BSRN FTP 用户名
BSRN_PASSWORD = "your_password"  # Replace with your BSRN FTP password / 替换为您的 BSRN FTP 密码

STATION = "QIQ" # Station abbreviation / 站点缩写
YEAR = 2024  # Now supports 4-digit integers / 现在支持 4 位整数
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "QIQ")

if __name__ == "__main__":
    if USERNAME == "your_username":
        print("Please edit 'download_qiq.py' to include your BSRN FTP credentials.")
    else:
        print(f"Starting download of all 2024 data for station {STATION.upper()}...")
        
        all_downloaded_paths = []
        
        # Using a single connection for each month batch is efficient
        # 每一个月的批量下载使用单个连接，这样很高效
        for month in range(1, 13):
            print(f"Fetching month {month:02d}...")
            # Demonstrating support for 4-digit year and mixed types
            # 展示对 4 位年份和混合类型的支持
            paths = download_bsrn_mon(
                stations=[STATION],
                year=YEAR,
                month=month,
                local_dir=LOCAL_DIR,
                username=USERNAME,
                password=PASSWORD
            )
            all_downloaded_paths.extend(paths)
        
        success_count = sum(1 for p in all_downloaded_paths if p is not None)
        print(f"\nDone! Downloaded {success_count}/12 files to: {LOCAL_DIR}")
