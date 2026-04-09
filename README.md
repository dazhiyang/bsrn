# bsrn

[![PyPI version](https://img.shields.io/pypi/v/bsrn.svg?v=0.2.1)](https://pypi.org/project/bsrn/)
[![Python Versions](https://img.shields.io/pypi/pyversions/bsrn.svg)](https://pypi.org/project/bsrn/)
[![Documentation Status](https://readthedocs.org/projects/bsrn/badge/?version=latest)](https://bsrn.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/bsrn)](https://pepy.tech/project/bsrn)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This **GitHub repository** is **`dazhiyang/bsrn`**: the source code and development tooling for the **`bsrn`** Python package.

**`bsrn`** is a community-developed toolbox that provides a set of robust functions and classes for processing and analyzing solar radiation data. The core mission of **`bsrn`** is to provide an open, reliable, interoperable, and benchmark-standard set of tools tailored specifically for the **Baseline Surface Radiation Network (BSRN)**. 

It features automated quality control (QC), high-precision solar geometry, clear-sky modeling, clear-sky detection (CSD), cloud enhancement event (CEE) detection, irradiance separation, and comprehensive data retrieval and visualization capabilities.

## 📖 [Documentation (Read the Docs)](https://bsrn.readthedocs.io)

## 🚀 Getting Started

### Installation

The core `bsrn` package is designed to be lightweight and fast. You can install it using pip:

**From PyPI (stable release):**
```bash
pip install bsrn
```

**From GitHub (latest development version):**
```bash
pip install git+https://github.com/dazhiyang/bsrn.git
```

### Optional Visualization Tools
If you want to use the built-in plotting features (like data availability charts or clear-sky calendars), you will need to install the optional visualization dependencies (plotnine, matplotlib, and scipy):

```bash
pip install bsrn[viz]
```

## Usage
For standard quality control and clear-sky modeling, simply import the base package:

```python
import bsrn

# Access core modules like bsrn.qc, bsrn.modeling, bsrn.io, bsrn.archive
```

If you installed the [viz] extra and want to generate plots, you must explicitly import the visualization submodule:

```python
import bsrn.visualization

# Access plotting tools like bsrn.visualization.calendar.plot_calendar()
```

### Quick Example — BSRNDataset (recommended)

```python
import bsrn

# One station-month from a BSRN LR0100 archive (.dat.gz)
ds = bsrn.BSRNDataset.from_file("data/QIQ/qiq0125.dat.gz")

# Typical pipeline (each step mutates the cached frame and returns it)
ds.solpos()                             # solar geometry + extraterrestrial
ds.clear_sky(model="rest2")             # ghi_clear / bni_clear / … (REST2; MERRA-2 via Hugging Face)
ds.qc_test()                            # flag columns: 0 = pass, 1 = fail
ds.qc_mask()                            # NaN failed irradiance; drop flag columns

df = ds.data()                          # minute-resolution table for analysis or export

# Visualize directly from the dataset (requires bsrn[viz])
ds.plot.daily("2025-01-15")             # UTC date inside the loaded month
ds.plot.table()                         # QC summary table
```

### Quick Example — Functional API

The same steps are available as standalone functions, useful for
non-BSRN data or custom DataFrames:

```python
from bsrn.io.retrieval import download_bsrn_stn, get_bsrn_file_inventory
from bsrn.physics.geometry import add_solpos_columns
from bsrn.modeling.clear_sky import add_clearsky_columns
from bsrn.qc.wrapper import run_qc

# 1. See what data is available
inventory = get_bsrn_file_inventory(["QIQ"], username="your_user", password="your_pass")

# 2. Download data for a station
download_bsrn_stn("QIQ", "data/QIQ", username="your_user", password="your_pass")

# 3. Read via BSRNDataset and get the DataFrame
ds = bsrn.BSRNDataset.from_file("data/QIQ/qiq0125.dat.gz")
df = ds.data()

# 4. Add solar position (recommended before time-averaging or clear-sky)
df = add_solpos_columns(df, "QIQ")

# 5. Add clear-sky reference columns (defaults to Ineichen)
df = add_clearsky_columns(df, "QIQ")

# 6. Run Quality Control (QC)
df = run_qc(df, "QIQ")

# 7. Add satellite-derived CAMS CRS all-sky columns
from bsrn.io.crs import add_crs_columns
df = add_crs_columns(df, "QIQ")

# 8. Visualize with plotnine
from bsrn.visualization.clearsky_models import plot_clearsky_models
plot_clearsky_models(df, "QIQ", date="2024-06-20", save_path="clearsky_qiq.pdf")
```

## 🛠 Features

The QC features, of which the implementation is primarily based on the [BSRN Operations Manual (2018)](https://bsrn.awi.de/) and [Forstinger et al. (2021)](https://doi.org/10.18086/swc.2021.36). See code for other references.

- **Level 1 (Physically Possible):** Absolute physical bounds for $G_h, B_n, D_h$, and $L_d$.
- **Level 2 (Extremely Rare):** Climatological limits for specific regimes.
- **Level 3 (Comparison):** Consistency checks ($G_h$ vs $B_n \cos Z + D_h$) with zenith-dependent thresholds.
- **Level 4 (Diffuse Ratio):** Diffuse-fraction and $k$–$k_t$ checks combining $G_h$, $D_h$, and extraterrestrial irradiance.
- **Level 5 (K-Indices):** Advanced clearness-index and $k_b$/$k_t$ index tests using clear-sky benchmarks and site elevation.
- **Level 6 (Tracker-Off Detection):** Identify tracking errors by comparing measured values with clear-sky and extraterrestrial irradiance.

Other important features include:

- **Solar Geometry:** Native NREL SPA implementation for high-precision solar position calculations.
- **Clear-Sky Models:** Ineichen (monthly Linke turbidity), McClear (CAMS SoDa API, from 2004 onward), and REST2 (MERRA-2 from Hugging Face).
- **Satellite Data:** Load CAMS solar radiation service (CRS) and National Solar Radiation Database (NSRDB) all-sky irradiance directly from Hugging Face into memory.
- **Clear-Sky Detection (CSD):** Reno, Ineichen, Lefevre, and BrightSun methods to identify clear-sky periods from irradiance time series.
- **Cloud Enhancement Event (CEE) Detection:** Killinger, Yang, and Gueymard methods to detect events when measured GHI significantly exceeds references.
- **Irradiance Separation:** Erbs, BRL, Engerer2, and Yang4 models to estimate diffuse fraction and DHI/BNI from GHI.
- **Robust Retrieval:** High-level API for FTP downloads from BSRN-AWI with exponential backoff retries (analysis functions assume **one station-to-archive file at a time**).
- **Station-to-archive formatting:** The `bsrn.archive` subpackage provides `LR_SPECS`, Fortran-style validators in `validation.py` (names referenced by each field’s `validate_func`), and ASCII output via `get_bsrn_format`. Scalar/header fields on the Pydantic `LR*` models use a single `lr_spec(lr_code, field_name, type, …)` annotation so metadata and post-parse checks stay in one place; LR0100/LR4000 minute columns use `field_validator` with `yearMonth` for vector length checks. Concrete types (`LR0001`–`LR4000CONST`) live in `records_models` and are re-exported from `bsrn.archive`; `get_azimuth_elevation` is in `archive_lr_formats` (also re-exported).
- **Visualization:** Data availability heatmaps and k vs kt separation plots via the very pretty `plotnine` (which reminds me of the good old R days).

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
│       ├── dataset.py                 # BSRNDataset: central monthly data object + pipeline methods
│       ├── constants.py               # Station database, Linke turbidity & physical constants
│       ├── archive/                   # Station-to-archive logical records (WRMC-style LR layouts)
│       │   ├── __init__.py            # Re-exports LR* models, LR_SPECS, get_azimuth_elevation, …
│       │   ├── specs.py               # LR_SPECS + station directory & A3–A7 code tables
│       │   ├── archive_lr_formats.py  # get_bsrn_format + get_azimuth_elevation (LR0004)
│       │   ├── records_base.py        # ArchiveRecordBase, make_archive_after_validator
│       │   ├── records_models.py      # lr_spec / lr_spec_field; LR0001–LR4000CONST Pydantic models
│       │   ├── formatting.py          # Fortran-style field formatting mixin
│       │   └── validation.py          # BSRN archive field validators (LR_SPECS validate_func)
│       ├── io/
│       │   ├── reader.py              # Read xxxmmyy.dat.gz station-to-archive files
│       │   ├── retrieval.py           # FTP downloads with retries
│       │   ├── merra2.py              # MERRA-2 parquet fetch (Hugging Face → RAM)
│       │   ├── mcclear.py             # SoDa McClear client helpers
│       │   ├── crs.py                 # SoDa CAMS solar radiation service (CRS) client helpers
│       │   ├── nsrdb.py               # NREL NSRDB all-sky data client helpers
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
│       │   └── cee_detection.py       # Cloud enhancement detection (Killinger, Yang, Gueymard)
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

# Automatically computes solar geometry if missing, but it is highly
# recommended to call `add_solpos_columns(df)` first for 1-minute data!
df = add_clearsky_columns(df, "QIQ")
# Adds columns: ghi_clear, bni_clear, dhi_clear
```

### Clear-Sky GHI from McClear (CAMS)

```python
from bsrn.modeling.clear_sky import add_clearsky_columns

# McClear data are available from 2004-01-01 onward.
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

### All-Sky GHI from NSRDB (NREL via Hugging Face)

Similar to REST2, NSRDB all-sky data is fetched directly from the Hugging Face dataset **[dazhiyang/bsrn-nsrdb-conus](https://huggingface.co/datasets/dazhiyang/bsrn-nsrdb-conus)** (and other variants).

```python
from bsrn.io.nsrdb import add_nsrdb_columns

# Fetch NSRDB all-sky GHI/DNI/DHI from Hugging Face
df = add_nsrdb_columns(df, station_code="QIQ", variant="conus")
# Adds columns: ghi_nsrdb, bni_nsrdb, dhi_nsrdb
```

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
# out_cee_*["is_enhancement"] is True/False/NA; out_cee_*["cee_flag"] is 0/1/NaN
```

### Data Availability Heatmap

```python
from bsrn.visualization.availability import plot_bsrn_availability

fig = plot_bsrn_availability(inventory_df, station_code="QIQ")
fig.save("availability.png", dpi=300)
```

### Station-to-archive logical records (`bsrn.archive`)

Logical records are **Pydantic v2** models (`LR0001`, …, `LR0100`, `LR4000`, `LR4000CONST`, …) defined in `records_models` and re-exported from `bsrn.archive`. The legacy umbrella type **`BSRNRecord` is removed**—use a concrete `LR*` model and call `get_bsrn_format` on the instance.

- **`LR_SPECS`** holds per-field `format`, missing tokens, defaults, and `validate_func` names.
- **Scalars:** validation runs through Pydantic `AfterValidator`, which calls the matching function in `bsrn.archive.validation`.
- **LR0100 / LR4000 minute vectors:** validators need `yearMonth`; those columns use a model-level `field_validator` instead.

```python
from bsrn.archive import LR0001, LR_SPECS

# Required keys for LR0001 are listed in LR_SPECS["LR0001"]
out = LR0001(stationNumber=94, month=1, year=2024, version=1).get_bsrn_format()
```

For minute blocks, pass `yearMonth="YYYY-MM"` and `pandas.Series` or `numpy.ndarray` per column (see `LR_SPECS["LR0100"]` / `["LR4000"]`), then `LR0100(...).get_bsrn_format(changed=True)` (and similarly for LR4000).

**Regression check (repository checkout):** from the repo root, generate a monthly `.dat` and compare to the checked-in reference (should match byte-for-byte):

```bash
PYTHONPATH=src python tests/2025-01/Code/2.station_to_archive.py \
  -o tests/2025-01/Output/qiq0125_run.dat --no-gzip
cmp tests/2025-01/Output/qiq0125_run.dat tests/2025-01/Output/qiq0125_ref.dat
```

Edit the `CONFIG` block at the top of `2.station_to_archive.py` for station-specific paths and metadata; the script expects the minute table at `tests/2025-01/Output/qiq0125.txt` for the default QIQ January 2025 example.

## Project standards

Development in this repository follows **`.agents/skills/project-rules/SKILL.md`**. In short:

- **Naming:** Use the radiometric **code names** from the skill (e.g. `ghi`, `bni`, `zenith`, `mu0`, `kt`, `Kt`, `ghi_extra`, `bni_extra`). In READMEs and technical docs, prefer **LaTeX-style symbols** (e.g. $G_h$, $B_n$, $k_t$) as there.
- **Documentation:** Public functions use **NumPy-style docstrings in English** (`Parameters`, `Returns`, `Raises` when applicable; `References` when based on literature). **Do not** use `->` return annotations on `def` lines; describe returns in the docstring.
- **BSRN data:** High-level workflows assume **one station archive file at a time** (e.g. one `XXXMMYY.dat.gz` per run); do not rely on silent multi-month concatenation inside the library.
- **Figures (when contributing plots):** Prefer **vector PDF** output, **Times New Roman** typography, and the **Wong** (discrete) / **Viridis** (continuous) palette conventions described in the skill; save generated figures to the **project root**, not under `src/` or `tests/`.

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
