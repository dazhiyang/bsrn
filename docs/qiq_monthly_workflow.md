# QIQ monthly BSRN workflow

This note summarizes the pipeline we set up: from raw 1 s inputs to a submission-ready **`.dat.gz`**, visual QC, human decisions recorded in **YAML**, and a controlled rewrite of the gzip archive.

Layout is **one folder per month**, named **`YYYY-MM`** (e.g. `2025-01`), containing **`Code/`** (scripts) and **`Output/`** (products).

---

## 1. From raw files to `.dat.gz`

| Step | Script | Role |
|------|--------|------|
| 1a | `Code/1.bsrn_1s_to_1min.py` | Reads BSRN-style 1 s **`.dat`** files, builds a full 1 Hz month grid, aggregates to **1-minute** columns, applies missing flags, writes a tab-separated **`.txt`** in `Output/` (name pattern from the script’s `output_filename`). |
| 1b | `Code/2.station_to_archive.py` | Reads that minute **`.txt`**, fills static logical records, uses **`bsrn.archive`** (`LR0100` / `LR4000`, `get_bsrn_format`) to emit BSRN station-to-archive **`.dat`**, and (by default) **`.dat.gz`** beside it under `Output/`. |

**Prerequisites**

- Month directory basename must match **`YYYY-MM`** (or pass `--yyyymm` explicitly in step 1).
- **`bsrn`** on `PYTHONPATH` or installed editable for steps 2–4.

**Typical commands** (from the month’s `Code/` directory, after activating the right Python env):

```bash
python 1.bsrn_1s_to_1min.py
python 2.station_to_archive.py
```

You should end up with something like `Output/qiq0125.dat` and `Output/qiq0125.dat.gz` (stem = station + MM + 2-digit year).

---

## 2. Visual screening (pre-submission checks)

| Script | Role |
|--------|------|
| `Code/3.presubmission_check.py` | Loads the **single production** `Output/*.dat.gz` (ignores `*_test.dat.gz`), infers **month** and **station stem** from the filename, maps `qiq` → registry code **QIQ**, runs solar geometry, daily QC stats, **`plot_qc_table`** → `*_qc_table.png`, and a multi-day **`plot_bsrn_timeseries_booklet`** → `*_timeseries.pdf`. |

**Discovery**

- No hand-edited `yyyymm` / `stn` in CONFIG for paths: the gzip basename drives the run.
- If **no** `.dat.gz` exists in `Output/`, the script exits immediately with a hint to create the gzip (e.g. `gzip -k` on the `.dat`).

**Dependencies**

- **`bsrn`** always; **plotnine** + **matplotlib** for the visualization imports.

Use the PNG/PDF for **visual screening** (spikes, tracker issues, frost periods, etc.) and decide what to flag for correction.

---

## 3. Human log: `masking_ledger.yaml`

Corrections are **not** hard-coded in Python. You record them in:

`Code/masking_ledger.yaml`

**Schema** (each item under `entries`):

- **`start_utc`**, **`end_utc`**: UTC range (minute resolution in practice).
- **`record`**: `lr0100` or `lr4000`.
- **`fields`**: list of names matching **`LR_SPECS`** in `bsrn` (e.g. `bni_avg`, `lwd_max`, `longwave_down`).
- **`reason`**: short justification (audit trail for you and reviewers).

**PyYAML** is required in the operator environment for the default ledger path (`pip install pyyaml`); it is **not** a dependency of the `bsrn` package. Alternatively, use a **`.json`** file with the same structure and `python 4.ad_hoc_correction.py --ledger your_ledger.json`.

Each month folder has its **own** `masking_ledger.yaml` unless you override `--ledger`.

---

## 4. Rewrite `.dat.gz` from the ledger

| Script | Role |
|--------|------|
| `Code/4.ad_hoc_correction.py` | Auto-discovers the same canonical **`Output/{stn}{MM}{YY2}.dat.gz`**, parses **LR0100** / **LR4000** data lines locally, applies the YAML masks (sets masked values to missing → **NaN**), regenerates those blocks with **`bsrn.archive`** (`LR0100` / `LR4000`, `get_bsrn_format`) so **`-999` / `-99.9` / `-99.99`** etc. match each field’s spec, splices them back into the full archive text, then **replaces** the production gzip after copying it to **`*.dat.gz.bak`**. Optional **`--output-gz PATH`** writes elsewhere instead. **`--dry-run`** parses and reports mask counts without writing. |

**Typical command**

```bash
python 4.ad_hoc_correction.py
# or: python 4.ad_hoc_correction.py --dry-run
```

---

## End-to-end order

1. **`1.bsrn_1s_to_1min.py`** → minute **`.txt`**
2. **`2.station_to_archive.py`** → **`.dat`** + **`.dat.gz`**
3. **`3.presubmission_check.py`** → plots + visual review
4. Edit **`masking_ledger.yaml`** with UTC windows, record, fields, reasons
5. **`4.ad_hoc_correction.py`** → updated **`.dat.gz`** (and **`.dat.gz.bak`** once)

Re-run step 3 after step 5 if you want plots against the final archive.

---

## Related files (per month)

| Path | Purpose |
|------|---------|
| `{YYYY-MM}/Code/masking_ledger.yaml` | Human-edited correction rules |
| `{YYYY-MM}/Output/*.dat.gz` | Canonical submission archive (single production file) |
| `{YYYY-MM}/Output/*.dat.gz.bak` | Backup created by `4.ad_hoc_correction.py` before overwrite |

---

*This document describes the QIQ workflow as implemented in the `Code/` scripts; it is not the BSRN format specification.*
