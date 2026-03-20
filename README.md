# bsrn

This **GitHub repository** is **`bsrn-qc`**: source code and development tooling for the **`bsrn`** Python package (published on [PyPI](https://pypi.org/project/bsrn/) from the same codebase).

`bsrn` is a Python package for the Baseline Surface Radiation Network (BSRN). It provides automated quality control (QC), solar geometry, clear-sky modeling, clear-sky detection (CSD), cloud enhancement event (CEE) detection, irradiance separation, data retrieval, and visualization tools for BSRN station-to-archive files.

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

From a local clone (editable install — edits under `src/bsrn/` take effect without reinstalling):
```bash
cd /path/to/bsrn-qc
pip install -e .
```

**Documentation:** [Read the Docs](https://bsrn.readthedocs.io)

### Quick Example (Single-File Workflow)

```python
from bsrn.io.retrieval import download_bsrn_stn, get_bsrn_file_inventory
from bsrn.io.readers import read_bsrn_station_to_archive
from bsrn.modeling.clear_sky import add_clearsky_columns
from bsrn.constants import BSRN_STATIONS

# 1. See what data is available
inventory = get_bsrn_file_inventory(["QIQ"], username="your_user", password="your_pass")

# 2. Download data for a station
download_bsrn_stn("QIQ", "data/QIQ", username="your_user", password="your_pass")

# 3. Read a single monthly file (one file at a time)
df = read_bsrn_station_to_archive("data/QIQ/qiq0124.dat.gz")

# 4. Add clear-sky reference columns for this file only
df = add_clearsky_columns(df, "QIQ")
```

## 🛠 Features

The QC implementation is primarily based on the [BSRN Operations Manual (2018)](https://bsrn.awi.de/) and [Forstinger et al. (2021)](https://doi.org/10.18086/swc.2021.36). See code for other references.

- **Level 1 (Physically Possible):** Absolute physical bounds for $G_h, B_n, D_h$, and $L_d$.
- **Level 2 (Extremely Rare):** Climatological limits for specific regimes.
- **Level 3 (Comparison):** Consistency checks ($G_h$ vs $B_n \cos Z + D_h$) with zenith-dependent thresholds.
- **Level 4 (Diffuse Ratio):** Diffuse-fraction and $k$–$k_t$ checks combining $G_h$, $D_h$, and extraterrestrial irradiance.
- **Level 5 (K-Indices):** Advanced clearness-index and $k_b$/$k_t$ index tests using clear-sky benchmarks and site elevation.
- **Level 6 (Tracker-Off Detection):** Identify tracking errors by comparing measured values with clear-sky and extraterrestrial irradiance.
- **Solar Geometry:** Native NREL SPA implementation for high-precision solar position calculations.
- **Clear-Sky Models:** Ineichen (monthly Linke turbidity), McClear (CAMS SoDa API, from 2004 onward), and REST2 (MERRA-2 from Hugging Face).
- **Clear-Sky Detection (CSD):** Reno, Ineichen, Lefevre, and BrightSun methods to identify clear-sky periods from irradiance time series.
- **Cloud Enhancement Event (CEE) Detection:** Killinger, Gueymard-style, and Wang methods to detect periods when measured GHI significantly exceeds clear-sky or extraterrestrial references and to filter events by temporal duration.
- **Irradiance Separation:** Erbs, BRL, Engerer2, and Yang4 models to estimate diffuse fraction and DHI/BNI from GHI.
- **Robust Retrieval:** High-level API for FTP downloads from BSRN-AWI with exponential backoff retries (analysis functions assume **one station-to-archive file at a time**).
- **Visualization:** Data availability heatmaps and k vs kt separation plots via `plotnine`.

## 📂 File Structure

> [!NOTE]
> Not all files are uploaded with Git. Data files and intermediate outputs are excluded via `.gitignore`.

```text
bsrn-qc/
├── pyproject.toml
├── LICENSE
├── README.md
├── .gitignore
├── .readthedocs.yaml              # Read the Docs build config
├── src/
│   └── bsrn/
│       ├── __init__.py
│       ├── constants.py               # Station database, Linke turbidity & physical constants
│       ├── io/
│       │   ├── readers.py             # Read .001, .002 station-to-archive files
│       │   ├── retrieval.py           # FTP downloads with retries
│       │   ├── merra2.py              # MERRA-2 parquet fetch (Hugging Face → RAM)
│       │   ├── mcclear.py             # SoDa McClear client helpers
│       │   ├── crs.py                 # SoDa CAMS solar radiation service (CRS) client helpers
│       │   └── writers.py             # Export results
│       ├── physics/
│       │   ├── spa.py                 # Native NREL SPA (solar position algorithm)
│       │   └── geometry.py            # Solar position and extraterrestrial irradiance
│       ├── qc/
│       │   ├── ppl.py                 # Physically possible limits (Level 1)
│       │   ├── erl.py                 # Extremely rare limits (Level 2)
│       │   ├── closure.py             # Internal consistency checks (Level 3)
│       │   ├── diff_ratio.py          # Diffuse ratio checks (Level 4)
│       │   ├── k_index.py             # Radiometric index tests (Level 5)
│       │   ├── tracker.py             # Solar tracker off detection (Level 6)
│       │   └── wrapper.py             # High-level QC pipeline
│       ├── visualization/
│       │   ├── availability.py        # File coverage heatmaps (plotnine)
│       │   ├── qc_table.py            # QC result tables
│       │   ├── separation.py          # Decomposition visualization
│       │   └── timeseries.py          # Time series plots
│       ├── utils/
│       │   ├── calculations.py        # Supporting math
│       │   ├── quality.py             # Quality utilities
│       │   ├── clear_sky_detection.py # Clear-sky detection (Reno, Ineichen, Lefevre, BrightSun)
│       │   └── cee_detection.py       # Cloud enhancement detection (Killinger, Gueymard, Wang)
│       └── modeling/
│           ├── clear_sky.py           # Ineichen clear-sky model
│           └── separation.py          # Irradiance separation (Erbs, BRL, Engerer2, Yang4)
├── docs/
│   ├── conf.py                        # Sphinx config; source dir = docs/ (tutorials + sphinx/ RST)
│   ├── index.rst                      # Site homepage (root index.html for Read the Docs)
│   ├── requirements.txt               # Sphinx / Read the Docs dependencies
│   ├── examples/                      # Examples landing page (index.rst) + optional scripts
│   │   └── index.rst
│   ├── tutorials/                     # Jupyter tutorials + index.rst (nbsphinx)
│   │   ├── 1.data_downloading.ipynb
│   │   ├── 2.quality_control.ipynb
│   │   ├── 3.time_averaging.ipynb
│   │   ├── 4.clear_sky_detection.ipynb
│   │   └── 5.cloud_enhancement_event.ipynb
│   └── sphinx/                        # RST (user_guide, api, _static); not the doc homepage
│       ├── api/                       # API reference (io, qc, physics, …)
│       └── user_guide/                # installation, getting_started, package_overview, …
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

### Clear-Sky GHI (Ineichen)

```python
from bsrn.modeling.clear_sky import add_clearsky_columns

df = add_clearsky_columns(df, "QIQ")
# Adds columns: ghi_clear, bni_clear, dhi_clear
```

### Clear-Sky GHI from McClear (CAMS)

```python
from bsrn.modeling.clear_sky import add_clearsky_columns

# McClear data are available from 2004-01-01 onward.
# McClear 数据自 2004-01-01 起可用。
df = add_clearsky_columns(
    df,
    station_code="QIQ",
    model="mcclear",
    mcclear_email="your_email@example.com",  # SoDa / CAMS account email
)
# Adds columns: ghi_clear, bni_clear, dhi_clear based on CAMS McClear
```

### Clear-Sky GHI from REST2 (MERRA-2 via Hugging Face)

REST2 uses MERRA-2 atmospheric inputs **only** from the Hugging Face dataset **[dazhiyang/bsrn-merra2](https://huggingface.co/datasets/dazhiyang/bsrn-merra2)** (hourly Parquet files per station, `station_code/*.parquet`). The `bsrn` package fetches them **into RAM** (no disk cache) when `model="rest2"` is used.

```python
from bsrn.modeling.clear_sky import add_clearsky_columns

# MERRA-2 is fetched from Hugging Face into RAM automatically.
df = add_clearsky_columns(df, station_code="QIQ", model="rest2")
# Adds columns: ghi_clear, bni_clear, dhi_clear based on REST2 + MERRA-2
```

The dataset README for Hugging Face is maintained in this repo at `data/bsrn_static_assets/README.md` (published to the Hub separately from PyPI).

### Clear-Sky Detection

```python
from bsrn.utils import detect_clearsky

# Requires GHI and clear-sky GHI (e.g. from add_clearsky_columns)
out = detect_clearsky("reno", ghi=df["ghi"], ghi_clear=df["ghi_clear"], times=df.index)
# out["is_clearsky"] is True/False/NA; out["cloud_flag"] is 0/1/NaN
# Other methods: "ineichen", "lefevre", "brightsun" (different inputs)
```

### Cloud Enhancement Event (CEE) Detection

```python
from bsrn.utils.cee_detection import detect_cee

# Killinger CEE detection: requires 1‑min GHI, clear-sky GHI, zenith, and a 1‑min index
out_cee_k = detect_cee(
    "killinger",
    ghi=df["ghi"],
    ghi_clear=df["ghi_clear"],
    zenith=df["zenith"],
    times=df.index,
)

# Gueymard-style CEE detection: flags kt = G_h / E_0 > 1
out_cee_g = detect_cee(
    "gueymard",
    ghi=df["ghi"],
    ghi_extra=df["ghi_extra"],
    times=df.index,
)

# Wang CEE detection: combines GHI, BNI, DHI masks and removes overly long events
out_cee_w = detect_cee(
    "wang",
    ghi=df["ghi"],
    ghi_clear=df["ghi_clear"],
    bni=df["bni"],
    bni_clear=df["bni_clear"],
    dhi=df["dhi"],
    dhi_clear=df["dhi_clear"],
    times=df.index,
    mag_threshold=1.10,
    bni_fraction=0.8,
    dhi_multiplier=1.5,
    max_duration_minutes=15.0,
)

# out_cee_*["is_enhancement"] is True/False/NA; out_cee_*["cee_flag"] is 0/1/NaN
```

### Data Availability Heatmap

```python
from bsrn.visualization.availability import plot_bsrn_availability

fig = plot_bsrn_availability(inventory_df, station_code="QIQ")
fig.save("availability.png", dpi=300)
```

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
