import os
import sys

# Add src to python path to ensure bsrn is importable
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from bsrn.visualization import plot_bsrn_availability
    from bsrn.visualization.timeseries import plot_bsrn_timeseries_booklet
    from bsrn.visualization.qc_table import plot_qc_table
    from bsrn.utils.quality import get_daily_stats
    import bsrn.io.readers as readers
    from bsrn.constants import BSRN_STATIONS
    from datetime import datetime
except ImportError:
    print("Error: Could not import bsrn. Make sure you are running from the project root.")
    sys.exit(1)

# Users can change these variables to their own directory / 用户可以根据自己的情况更改目录
# --- CONFIGURATION ---
BSRN_USER = "your_username" # Replace with your BSRN FTP username / 替换为您的 BSRN FTP 用户名
BSRN_PASSWORD = "your_password"  # Replace with your BSRN FTP password / 替换为您的 BSRN FTP 密码

# List of stations you want to check
STATIONS_TO_CHECK = ['PAY', 'NYA', 'GVN', 'ILO', 'TAT', 'QIQ'] 

# Year range
START_YEAR = 1992

# Test for station QIQ, December 2024
FILE_PATH = "data/QIQ/qiq1224.dat.gz"

def test_availability():
    print(f"\n--- Testing Availability Heatmap ---")
    print(f"Initiating BSRN FTP search for: {STATIONS_TO_CHECK}")
    
    try:
        output_file = "bsrn_availability_heatmap.pdf"
        plot_bsrn_availability(
            stations=STATIONS_TO_CHECK,
            username=BSRN_USER,
            password=BSRN_PASSWORD,
            start_year=START_YEAR,
            output_file=output_file
        )
        print(f"Successfully generated heatmap: {output_file}")
    except Exception as e:
        print(f"Availability heatmap failed: {e}")

def test_timeseries():
    print(f"\n--- Testing Timeseries Booklet ---")
    
    if not os.path.exists(FILE_PATH):
        print(f"Skipping timeseries test: {FILE_PATH} not found.")
        return

    try:
        output_file = "test_qiq_booklet.pdf"
        print(f"Generating booklet for {FILE_PATH}...")
        plot_bsrn_timeseries_booklet(
            FILE_PATH, 
            output_file, 
            station_code="QIQ", 
            apply_qc=False
        )
        print(f"Successfully generated booklet: {output_file}")
    except Exception as e:
        print(f"Timeseries booklet failed: {e}")

def test_qc_table():
    print(f"\n--- Testing QC Table Heatmap ---")
    
    if not os.path.exists(FILE_PATH):
        print(f"Skipping QC table test: {FILE_PATH} not found.")
        return

    try:
        output_file = "test_qiq_qc_table.pdf"
        print(f"Reading {FILE_PATH}...")
        df = readers.read_bsrn_station_to_archive(FILE_PATH)
        if df is not None:
            station_code = "QIQ"
            meta = BSRN_STATIONS[station_code]
            print("Calculating QC stats...")
            daily_stats = get_daily_stats(df, meta["lat"], meta["lon"], meta["elev"])
            
            filename = os.path.basename(FILE_PATH)
            month_label = datetime.strptime(filename[3:7], "%m%y").strftime("%B %Y")
            title = f"BSRN quality flags and sunshine duration for {month_label}"
            
            print(f"Generating plot: {output_file}...")
            plot_qc_table(daily_stats, title, output_file)
            print(f"Successfully generated QC table: {output_file}")
    except Exception as e:
        print(f"QC table plot failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    # test_availability()
    # test_timeseries()
    test_qc_table()

if __name__ == "__main__":
    main()
