---
name: bsrn-project-rules
description: Standards and governance for the bsrn project, including naming conventions and documentation rules.
---

# 🤖 Project Standards & Governance

This document defines the naming conventions, scientific symbols, and documentation standards used throughout the `bsrn` project.

**Contributors and automated agents** MUST read this skill **before** editing project code and **apply** the rules below (especially §Documentation — function signatures and docstrings). Treat it as binding, not background.

## 📏 Radiometric Parameters

Code variables and documentation **MUST** strictly adhere to the following table:

| Acronym | Code Name | Symbol | Full Name |
| :--- | :--- | :--- | :--- |
| **GHI** | `ghi` | $G_h$ | global horizontal irradiance |
| **BNI** | `bni` | $B_n$ | beam normal irradiance |
| **DHI** | `dhi` | $D_h$ | diffuse horizontal irradiance |
| **LWD** | `lwd` | $L_d$ | downward longwave radiation |
| **LWU** | `lwu` | $L_u$ | upward longwave radiation |
| **SWU** | `swu` | $S_u$ | upward shortwave radiation |
| **NET** | `net` | $R_n$ | net radiation |
| **SZA** | `zenith` | $Z$ | solar zenith angle |
| **cosSZA** | `mu0` | $\mu_0$ | cosine of SZA |
| **SAA** | `azimuth` | $\phi$ | solar azimuth angle |
| **GHIC** | `ghi_clear` | $G_{hc}$ | clear-sky GHI |
| **BNIC** | `bni_clear` | $B_{nc}$ | clear-sky BNI |
| **DHIC** | `dhi_clear` | $D_{hc}$ | clear-sky DHI |
| **LWDC** | `lwd_clear` | $L_{dc}$ | clear-sky LWD |
| **LWUC** | `lwu_clear` | $L_{uc}$ | clear-sky LWU |
| **BNIE** | `bni_extra` | $E_{0n}$ | extraterrestrial BNI |
| **GHIE** | `ghi_extra` | $E_{0}$ | extraterrestrial GHI |
| **SC** | `solar_constant` | $E_{\text{sc}}$ | solar constant |
| **CSI** | `kappa` | $\kappa$ | clear-sky index |
| **k_t** | `kt` | $k_t$ | clearness index |
| **K_t** (daily) | `Kt` | $K_t$ | daily clearness index |
| **k_b** | `kb` | $k_b$ | beam transmittance |
| **k_d** | `kd` | $k_d$ | diffuse transmittance |
| **k** | `k` | $k$ | diffuse fraction |
| **TMP** | `temp` | $T$ | air temperature |
| **RH** | `rh` | $RH$ | relative humidity |
| **SP** | `pressure` | $P$ | station pressure |

---

## 🚩 Quality Control Flags
- For the data points that do not pass **PPL test**, we **MUST** use the following flag names:
    - `flagPPLGHI`: Flag for GHI physically possible limit test
    - `flagPPLBNI`: Flag for BNI physically possible limit test
    - `flagPPLDHI`: Flag for DHI physically possible limit test
    - `flagPPLLWD`: Flag for LWD physically possible limit test
- For the data points that do not pass **ERL test**, we **MUST** use the following flag names:
    - `flagERLGHI`: Flag for GHI extremely rare limit test
    - `flagERLBNI`: Flag for BNI extremely rare limit test
    - `flagERLDHI`: Flag for DHI extremely rare limit test
    - `flagERLLWD`: Flag for LWD extremely rare limit test
- For the data points that do not pass **Closure test**, we **MUST** use the following flag names:
    - `flag3lowSZA`: Flag for closure test at low SZA ($Z \le 75^\circ$)
    - `flag3highSZA`: Flag for closure test at high SZA ($Z > 75^\circ$)
- For the data points that do not pass **Diffuse ratio (k) test**, we **MUST** use the following flag names:
    - `flagKKt`: Flag for combined $k$ and $k_t$ test ($k < 0.96$)
    - `flagKlowSZA`: Flag for diffuse ratio test at low SZA ($Z < 75^\circ$, $k < 1.05$)
    - `flagKhighSZA`: Flag for diffuse ratio test at high SZA ($Z \ge 75^\circ$, $k < 1.1$)
- For the data points that do not pass **k-index test**, we **MUST** use the following flag names:
    - `flagKbKt`: Flag for $k_b < k_t$ test
    - `flagKb`: Flag for $k_b$ physical limit test
    - `flagKt`: Flag for $k_t$ physical limit test
- For the data points that do not pass **Tracker-off test**, we **MUST** use the following flag name:
    - `flagTracker`: Flag for solar tracker failure detection

---

## 📝 Documentation & Coding Rules

> ### **REQUIRED — Public function docstrings (NumPy style)**
>
> Every **function** and **public method** (including module-level callables and class methods that form the API) **MUST** use NumPy-style docstrings in English (see **Docstring Structure** below).
>
> **Required sections (in order):**
>
> 1. **Summary** — concise English description.
> 2. **`Parameters`** — always present if the callable has parameters (use `None` / optional wording where needed).
> 3. **`Returns`** — always present unless the callable returns `None` *and* that is obvious; prefer an explicit ``None`` description when the function exists only for side effects.
> 4. **`Raises`** — **whenever** the body can raise a documented exception (typically `ValueError`, `TypeError`, `KeyError`, etc.). Omit the section only if truly no exception is part of the contract.
>
> **`References`** — still **MUST** appear when the implementation follows a paper or external spec (see §2).
>
> Data-only modules (e.g. large static dicts with no functions) are exempt from per-function docstrings.

> ### **REQUIRED — Function signature layout (no `->`, wrap at 80)**
>
> - **MUST NOT** use PEP 484 **return** annotations on definitions: do **not** write `-> SomeType` after the closing `)` of `def` (e.g. forbid `def f(x) -> str:`). Describe return types in the **`Returns`** section of the docstring instead.
> - **MUST** keep each **line of the `def` signature** to **≤ 80 characters** (indent + `def` name + parameters). If wrapping is needed, **continue on the next line** and **group several parameters per line** so that **each line stays ≤ 80 characters**.
> - **MUST NOT** end the first signature line immediately after `(` with **no parameters** on that line — i.e. do **not** write:
>   ```python
>   def long_function_name(
>       arg1, arg2, ...
>   ):
>   ```
>   **MUST** break **after a comma** so the opening line still contains parameters, then align the continuation with the `(` that opens the parameter list (first parameter on the next line starts in the same column as parameters on the first line):
>   ```python
>   def long_function_name(arg1, arg2, arg3,
>                          arg4, arg5):
>   ```
> - **MUST NOT** default to **one parameter per line** (vertical “arg list”) when a compact multi-parameter line still fits the 80-character rule. Use a vertical list only when unavoidable (e.g. many parameters or long default expressions), still grouping parameters to minimize line count.
> - The same **80-character, grouped** idea applies to **long function calls** when you must break them across lines: avoid one argument per line unless necessary; prefer breaking after a comma with **multiple arguments on the continuation line** where possible.

### 1. Docstring Structure
- **MUST** use NumPy/SciPy style with English descriptions.
- **MUST** include English in the summary and parameter descriptions.
- **MUST** include **`Parameters`**, **`Returns`**, and **`Raises`** (when applicable); see the **highlighted box above**.
- **MUST** include a `References` section at the end of the docstring for functions based on literature.

```python
def function_name(param):
    """
    One-line summary of what the function does.

    Parameters
    ----------
    param : type
        Description of param.

    Returns
    -------
    result : type
        Description of return value.

    Raises
    ------
    ValueError
        When validation fails.

    References
    ----------
    .. [1] Author, A. (Year). Title. Journal, Vol(Issue), Pages.
    """
```

### 2. Citations
- **MUST** follow **APA style** for all citations (e.g. Author, A. (Year). Title. Journal, Vol(Issue), Pages).
- **MUST** use the reStructuredText citation format (`.. [1]`) within the docstring's `References` section.
- **MUST** use sentence case for scientific paper titles in references.

### 3. Naming & Consistency
- **MUST** use the exact **Code Name** from the Radiometric Parameters table above for **all** local variables, function parameters, and intermediate results. The table is the single source of truth.
- **DO NOT** invent synonyms or pvlib-style aliases. Common mistakes and their **mandatory** replacements:

| Forbidden Name | Correct Code Name | Why |
| :--- | :--- | :--- |
| `cos_zenith`, `cos_z`, `cosz` | `mu0` | Table row **cosSZA** |
| `i0h`, `I0h`, `i0`, `etr_h` | `ghi_extra` | Table row **GHIE** |
| `dni_extra`, `etr_n`, `I0n` | `bni_extra` | Table row **BNIE** |
| `solar_zenith`, `sza`, `sz` | `zenith` | Table row **SZA** |
| `solar_azimuth`, `saa` | `azimuth` | Table row **SAA** |
| `K_t`, `k_t`, `clearness` (hourly) | `kt` | Table row **k_t** |
| `K_t_daily`, `K_t` (daily variable) | `Kt` | Table row **K_t** (daily) |
| `df` (diffuse fraction) | `k` | Table row **k** |

- **DO NOT** use generic terms like "Direct" or "Global" in isolation.
- **MUST** use LaTeX symbols ($G_h, B_n$, etc.) in READMEs and technical docs.
- **DO NOT** capitalize the long form of abbreviations (e.g., use "global horizontal irradiance").
- **MUST** limit general line length to a maximum of **110 characters** (docstrings and comments included), **except** that **`def` signature lines** follow the **80-character** rule in the highlighted box above.
- **MUST** use sentence case for scientific paper titles in references.

### 4. Function Signature Style
- **MUST** follow the **highlighted box** in §Documentation: **no `->` return annotations**, **≤ 80 characters per signature line**, **group parameters** on wrapped lines (avoid one-parameter-per-line unless necessary).
- **MUST** use compact signatures: required parameters first, then optional parameters with defaults.
- If wrapping is needed, **MUST** split into **a few short lines**: group several parameters per line so **each line stays ≤ 80 characters** for the `def` line(s).
- For continuation lines after a trailing comma, **MUST** indent continuation lines consistently with this repo (see `run_qc` in `bsrn/qc/wrapper.py` or `pretty_average` in `bsrn/utils/averaging.py` where 110-char wrapping was used historically; new code should prefer **80** for `def` lines as above).
- Example (short, one line): `def add_clearsky_columns(df, station_code=None, lat=None, lon=None, elev=None, model="ineichen"):`
- Example (wrapped, two lines, ≤80 chars each): break after a comma so the next line holds **multiple** parameters, not one per line; **do not** put only `def name(` on the first line (see highlighted box above).

### 5. Output Directory
- All generated plots and figures **MUST** be saved to the project root directory.
- **DO NOT** save output files into `tests/`, `src/`, or other source directories.

### 6. Git Usage
- **DO NOT** push to git unless explicitly instructed by the USER.
- **MUST** exclude all PDF files (`*.pdf`) from Git to avoid bloat in the repository.

### 7. BSRN File Handling Policy
- **Single-file workflow**: All high-level workflows (QC, clear-sky modeling, CSD, separation, visualization) **MUST** operate on **one BSRN station-to-archive file at a time** (for example, a single `XXXMMYY.dat.gz` monthly file).
- **No implicit concatenation**: Library functions **MUST NOT** silently concatenate multiple months or years internally. If users need multi-month analyses, they **MUST** loop over files and combine results explicitly at the application level.
- **Index scope**: Within a single run, functions **MUST** assume that the `DatetimeIndex` comes from a single contiguous BSRN monthly file; cross-file or cross-year assumptions are out of scope for the core package.

### 8. Visualization & Aesthetics
- **Color Palettes**:
    - **Discrete Variables**: **MUST** use the **Wong colorblind-friendly palette**. Colors **MUST** be used in this specific order based on the number of categories:
        1. `#E69F00` (Orange)
        2. `#56B4E9` (Sky Blue)
        3. `#009E73` (Bluish Green)
        4. `#CC79A7` (Reddish Purple)
        5. `#D55E00` (Vermillion)
        6. `#F0E442` (Yellow)
        7. `#0072B2` (Blue)
    - **Continuous Variables**: **MUST** use the **Viridis** color palette (or equivalent perceptually uniform colormaps).
- **Line Size**: Default line size **MUST** be set to 0.3 for all plots.
- **Fonts**: **MUST** use 'Times New Roman' for all axis labels, titles, and legends.
- **Size**:
    - Figure width **MUST** be **160 mm** where a fixed journal column width applies.
    - **9 pt** for all text (titles, axes, legend, cell labels) in **faceted / tabular** plotnine figures: `timeseries.py` (day & booklet) and `qc_table.py` (`plot_qc_table`).
    - **7 pt** remains the default for simpler layouts (e.g. `availability.py`, `calendar.py`) unless a figure is explicitly aligned with the 9 pt stack above.
- **Format**: All plots **MUST** be output in **PDF** format to ensure high-quality vector graphics.
- **Color bar (continuous legend)**:
    - **MUST** match the setup used in `availability.py` for consistency across plots.
    - Use `scale_fill_cmap(cmap_name='viridis', name="<Legend title>")` for continuous fill scales.
    - In `theme()`: `legend_position="bottom"`, `legend_title=element_text(size=7)`, `legend_text=element_text(size=7)`, `legend_key_width=100`, `legend_key_height=5`, `legend_margin=-12`, `legend_box_spacing=0`. To add space between x-axis title and legend without shifting the legend, use `axis_title_x=element_text(size=7, margin={"b": 8})` (bottom margin in points). Use **9 pt** in those `element_text` calls when editing **timeseries** legends to match §Size above.
    - **Plot margin** (optional): `plot_margin_top`, `plot_margin_right`, `plot_margin_bottom`, `plot_margin_left` (each in `[0, 1]`; use 0 for no extra margin).

