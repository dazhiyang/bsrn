# QIQ monthly BSRN workflow

> [!NOTE]
> **Purpose** — Pipeline from raw 1 s inputs to a submission-ready **`.dat.gz`**, visual QC, human decisions in **YAML**, and a controlled rewrite of the gzip archive.
>
> **Layout** — One folder per month: **`YYYY-MM`** (e.g. `2025-01`), with **`Code/`** (scripts) and **`Output/`** (products).

On **GitHub.com**, the colored callouts below use built-in alert styles (blue / green / yellow / red accent). In plain viewers they appear as quoted blocks.

---

## 1 · From raw files to `.dat.gz`

| Step | Script | Role |
|:----:|--------|------|
| **1a** | `Code/1.bsrn_1s_to_1min.py` | Reads BSRN-style 1 s **`.dat`** files, builds a full 1 Hz month grid, aggregates to **1-minute** columns, applies missing flags, writes a tab-separated **`.txt`** in `Output/`. |
| **1b** | `Code/2.station_to_archive.py` | Reads that minute **`.txt`**, fills static logical records, uses **`bsrn.archive`** (`LR0100` / `LR4000`, `get_bsrn_format`) to emit **`.dat`** and (by default) **`.dat.gz`** under `Output/`. |

> [!IMPORTANT]
> **Prerequisites**
> - Month directory basename must match **`YYYY-MM`** (or pass `--yyyymm` in step 1).
> - **`bsrn`** on `PYTHONPATH` or installed editable for steps 2–4.

> [!TIP]
> **Run from the month’s `Code/` folder** (correct Python env active):
> ```bash
> python 1.bsrn_1s_to_1min.py
> python 2.station_to_archive.py
> ```

> [!NOTE]
> **Expected outputs** — e.g. `Output/qiq0125.dat` and `Output/qiq0125.dat.gz` (stem = station + MM + 2-digit year).

---

## 2 · Visual screening (pre-submission)

| Script | Role |
|--------|------|
| `Code/3.presubmission_check.py` | Loads the **single production** `Output/*.dat.gz` (ignores `*_test.dat.gz`), infers month and station stem from the filename, maps `qiq` → **QIQ**, runs solar geometry, daily QC stats, **`plot_qc_table`** → `*_qc_table.png`, and **`plot_bsrn_timeseries_booklet`** → `*_timeseries.pdf`. |

> [!NOTE]
> **Discovery** — No hand-edited `yyyymm` / `stn` in CONFIG: the gzip basename drives the run. If **no** `.dat.gz` exists in `Output/`, the script exits with a hint (e.g. `gzip -k` on the `.dat`).

> [!IMPORTANT]
> **Dependencies** — **`bsrn`** always; **plotnine** + **matplotlib** for plots.

Use the **PNG** / **PDF** for visual screening (spikes, tracker issues, frost, etc.), then record corrections in the ledger (§3).

---

## 3 · Human log: `masking_ledger.yaml`

Corrections are **not** hard-coded in Python. Record them in **`Code/masking_ledger.yaml`**.

> [!NOTE]
> **Schema** (each item under `entries`):
> - **`start_utc`**, **`end_utc`** — UTC range (minute resolution in practice).
> - **`record`** — `lr0100` or `lr4000`.
> - **`fields`** — Names matching **`LR_SPECS`** (e.g. `bni_avg`, `lwd_max`, `longwave_down`).
> - **`reason`** — Short justification (audit trail).

> [!TIP]
> **PyYAML** — `pip install pyyaml` in the operator env (not a core `bsrn` dependency). Or use a **`.json`** ledger: `python 4.ad_hoc_correction.py --ledger your_ledger.json`.

Each month folder has its **own** `masking_ledger.yaml` unless you pass **`--ledger`**.

---

## 4 · Rewrite `.dat.gz` from the ledger

| Script | Role |
|--------|------|
| `Code/4.ad_hoc_correction.py` | Discovers canonical **`Output/{stn}{MM}{YY2}.dat.gz`**, parses **LR0100** / **LR4000** lines, applies YAML masks (missing → **NaN**), regenerates blocks with **`bsrn.archive`** so **`-999` / `-99.9` / `-99.99`** match each field’s spec, splices back into the full archive, **replaces** the production gzip after copying to **`*.dat.gz.bak`**. Optional **`--output-gz`**, **`--dry-run`**. |

> [!WARNING]
> **Writes & backup** — Production **`.dat.gz`** is overwritten after a **`.bak`** copy. Use **`--dry-run`** first to see mask counts without writing.

```bash
python 4.ad_hoc_correction.py
# or:
python 4.ad_hoc_correction.py --dry-run
```

---

## End-to-end order

| # | Step | Output |
|:-:|------|--------|
| 1 | **`1.bsrn_1s_to_1min.py`** | Minute **`.txt`** |
| 2 | **`2.station_to_archive.py`** | **`.dat`** + **`.dat.gz`** |
| 3 | **`3.presubmission_check.py`** | Plots → visual review |
| 4 | Edit **`masking_ledger.yaml`** | UTC windows, record, fields, reasons |
| 5 | **`4.ad_hoc_correction.py`** | Updated **`.dat.gz`** + **`.dat.gz.bak`** |

> [!TIP]
> Re-run **step 3** after step 5 if you want plots against the final archive.

---

## Related paths (per month)

| Path | Purpose |
|------|---------|
| `{YYYY-MM}/Code/masking_ledger.yaml` | Human-edited correction rules |
| `{YYYY-MM}/Output/*.dat.gz` | Canonical submission archive |
| `{YYYY-MM}/Output/*.dat.gz.bak` | Backup from `4.ad_hoc_correction.py` |

---

> [!CAUTION]
> This document describes the **QIQ** workflow implemented in the month **`Code/`** scripts; it is **not** the official BSRN format specification.
