# bsrn

`bsrn` is a Python package for the Baseline Surface Radiation Network (BSRN). It provides automated quality control (QC), solar geometry, clear-sky modeling, data retrieval, and visualization tools for BSRN station-to-archive files.

## 🚀 Getting Started

### Installation

From PyPI (stable release):
```bash
pip install bsrn
```

From GitHub (latest development version):
```bash
pip install git+https://github.com/dazhiyang/bsrn.git
```

### Quick Example

```python
from bsrn.io.retrieval import download_bsrn_stn, get_bsrn_file_inventory
from bsrn.io.readers import read_bsrn_station_to_archive
from bsrn.physics.clearsky import add_clearsky_columns
from bsrn.constants import BSRN_STATIONS

# 1. See what data is available
inventory = get_bsrn_file_inventory(["QIQ"], username="your_user", password="your_pass")

# 2. Download data for a station
download_bsrn_stn("QIQ", "data/QIQ", username="your_user", password="your_pass")

# 3. Read a monthly file
df = read_bsrn_station_to_archive("data/QIQ/qiq0124.dat.gz")

# 4. Add clear-sky reference columns
df = add_clearsky_columns(df, "QIQ")
```

## 🛠 Features

Based on the [BSRN Operations Manual (2018)](https://bsrn.awi.de/) and [Forstinger et al. (2021)](https://doi.org/10.18086/swc.2021.36):

- **Level 1 (Physically Possible):** Absolute physical bounds for $G_h, B_n, D_h$, and $L_d$.
- **Level 2 (Extremely Rare):** Climatological limits for specific regimes.
- **Level 3 (Comparison):** Consistency checks ($G_h$ vs $B_n \cos Z + D_h$) with zenith-dependent thresholds.
- **Radiometric Indices:** Advanced checks using clearness index ($k_t$), beam transmittance ($k_b$), and diffuse fraction ($k$).
- **Tracker Detection:** Identify tracking errors by comparing measured values with clear-sky benchmarks.
- **Solar Geometry:** Native NREL SPA implementation for high-precision solar position calculations.
- **Clear-Sky Models:** Ineichen model with monthly Linke turbidity for all BSRN stations.
- **Robust Retrieval:** High-level API for batch FTP downloads from BSRN-AWI with exponential backoff retries.
- **Visualization:** Data availability heatmaps via `plotnine`.

## 📂 File Structure

> [!NOTE]
> Not all files are uploaded with Git. Data files and intermediate outputs are excluded via `.gitignore`.

```text
bsrn-qc/
├── pyproject.toml
├── LICENSE
├── README.md
├── .gitignore
├── src/
│   └── bsrn/
│       ├── __init__.py
│       ├── constants.py          # Station database, Linke turbidity & physical constants
│       ├── io/
│       │   ├── readers.py        # Read .001, .002 station-to-archive files
│       │   ├── retrieval.py      # FTP downloads with retries
│       │   └── writers.py        # Export results
│       ├── physics/
│       │   ├── spa.py            # Native NREL SPA (solar position algorithm)
│       │   ├── geometry.py       # Solar position and extraterrestrial irradiance
│       │   └── clearsky.py       # Ineichen clear-sky model
│       ├── qc/
│       │   ├── ppl.py            # Physically possible limits (Level 1)
│       │   ├── erl.py            # Extremely rare limits (Level 2)
│       │   ├── closure.py        # Internal consistency checks (Level 3)
│       │   ├── k_index.py        # Radiometric index tests
│       │   ├── diff_ratio.py     # Diffuse ratio checks
│       │   ├── tracker.py        # Solar tracker status detection
│       │   └── wrapper.py        # High-level QC pipeline
│       ├── visualization/
│       │   ├── availability.py   # File coverage heatmaps (plotnine)
│       │   ├── qc_table.py       # QC result tables
│       │   ├── separation.py     # Decomposition visualization
│       │   └── timeseries.py     # Time series plots
│       ├── utils/
│       │   ├── calculations.py   # Supporting math
│       │   └── quality.py        # Quality utilities
│       └── modeling/
│           └── separation.py     # Solar radiation separation (Engerer2, Yang4)
├── tests/
│   ├── test_io.py
│   ├── test_physics.py
│   ├── test_visualization.py
│   └── test_modeling.py
├── notebooks/
│   └── qc_demo.ipynb             # QC workflow demo
└── data/
    ├── download_qiq.py           # Script to download QIQ station data
    ├── retrive_Linke.py          # Script to retrieve Linke turbidity data
    └── QIQ/                      # Sample data for station QIQ
```

## 📖 Examples

### Solar Position

```python
import pandas as pd
from bsrn.physics.geometry import get_solar_position, get_bni_extra

times = pd.date_range("2024-07-01", periods=1440, freq="1min", tz="UTC")
solpos = get_solar_position(times, lat=47.80, lon=124.49, elev=170)

print(solpos[["zenith", "apparent_zenith", "azimuth"]].head())
```

### Extraterrestrial Irradiance

```python
from bsrn.physics.geometry import get_bni_extra

bni_extra = get_bni_extra(times)  # Spencer (1971) method
```

### Clear-Sky GHI

```python
from bsrn.physics.clearsky import add_clearsky_columns

df = add_clearsky_columns(df, "QIQ")
# Adds columns: ghi_clear, bni_clear, dhi_clear
```

### Data Availability Heatmap

```python
from bsrn.visualization.availability import plot_bsrn_availability

fig = plot_bsrn_availability(inventory_df, station_code="QIQ")
fig.save("availability.png", dpi=300)
```

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
