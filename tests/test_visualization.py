import os
import sys

# Add src to python path to ensure bsrn is importable
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from bsrn.visualization import plot_bsrn_availability, plot_k_vs_kt
    from bsrn.visualization.timeseries import plot_bsrn_timeseries_booklet
    from bsrn.visualization.qc_table import plot_qc_table
    from bsrn.utils.quality import get_daily_stats
    from bsrn.physics.clearsky import add_clearsky_columns
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
# Directory for full-year test: place 12 monthly files (qiq0124.dat.gz ... qiq1224.dat.gz) in data/QIQ
DATA_DIR_YEAR = "data/QIQ"

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
            station_name = meta["name"]
            print("Calculating QC stats...")
            daily_stats = get_daily_stats(df, meta["lat"], meta["lon"], meta["elev"])
            
            filename = os.path.basename(FILE_PATH)
            month_label = datetime.strptime(filename[3:7], "%m%y").strftime("%B %Y")
            title = f"BSRN quality flags and sunshine duration for {station_code}, {month_label}"
            
            print(f"Generating plot: {output_file}...")
            plot_qc_table(daily_stats, title, output_file)
            print(f"Successfully generated QC table: {output_file}")
    except Exception as e:
        print(f"QC table plot failed: {e}")
        import traceback
        traceback.print_exc()


def test_separation_k_vs_kt():
    """Test k vs kt plot: one row of facets, all four models; use a year of data if available."""
    print(f"\n--- Testing Separation k vs kt Plot ---")

    # Prefer full year from DATA_DIR_YEAR; fallback to single file FILE_PATH
    df = None
    if os.path.isdir(DATA_DIR_YEAR):
        df = readers.read_bsrn_multiple_files(DATA_DIR_YEAR, extension="qiq*.dat.gz")
    if df is None or len(df) == 0:
        if os.path.exists(FILE_PATH):
            df = readers.read_bsrn_station_to_archive(FILE_PATH)
        else:
            print(f"Skipping separation plot test: neither {DATA_DIR_YEAR} nor {FILE_PATH} found.")
            return

    if df is None or "ghi" not in df.columns or "dhi" not in df.columns:
        print("Skipping: no GHI/DHI in data.")
        return

    try:
        station_code = "QIQ"
        meta = BSRN_STATIONS[station_code]
        lat, lon = meta["lat"], meta["lon"]
        df = add_clearsky_columns(df, station_code=station_code)
        from bsrn.modeling import engerer2_separation, yang4_separation
        res_e2 = engerer2_separation(
            df.index, df["ghi"].values, lat, lon,
            ghi_clear=df["ghi_clear"].values
        )
        res_y4 = yang4_separation(
            df.index, df["ghi"].values, lat, lon,
            ghi_clear=df["ghi_clear"].values
        )
        df = df.copy()
        df["k_engerer2"] = res_e2["k"].reindex(df.index).values
        df["k_yang4"] = res_y4["k"].reindex(df.index).values

        plot_k_vs_kt(
            df, models=("erbs", "brl", "engerer2", "yang4"), lat=lat, lon=lon,
            k_mod_cols={"engerer2": "k_engerer2", "yang4": "k_yang4"},
            output_file="k_vs_kt_facet.pdf"
        )
        print(f"Successfully generated k_vs_kt_facet.pdf (n={len(df)} points)")
    except Exception as e:
        print(f"Separation k vs kt plot failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    # test_availability()
    # test_timeseries()
    # test_qc_table()
    test_separation_k_vs_kt()

if __name__ == "__main__":
    main()
