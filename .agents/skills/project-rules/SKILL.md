---
name: bsrn-project-rules
description: Standards and governance for the bsrn project, including naming conventions and documentation rules.
---

# 🤖 Project Standards & Governance

This document defines the naming conventions, scientific symbols, and bilingual (English/Chinese) translations used throughout the `bsrn` project.

## 📏 Radiometric Parameters

Code variables and documentation **MUST** strictly adhere to the following table:

| Acronym | Code Name | Symbol | Full English Name | Full Chinese Name |
| :--- | :--- | :--- | :--- | :--- |
| **GHI** | `ghi` | $G_h$ | global horizontal irradiance | 水平总辐照度 |
| **BNI** | `bni` | $B_n$ | beam normal irradiance | 法向直接辐照度 |
| **DHI** | `dhi` | $D_h$ | diffuse horizontal irradiance | 水平散射辐照度 |
| **LWD** | `lwd` | $L_d$ | downward longwave radiation | 下行长波辐射 |
| **LWU** | `lwu` | $L_u$ | upward longwave radiation | 上行长波辐射 |
| **SWU** | `swu` | $S_u$ | upward shortwave radiation | 上行短波辐射 |
| **NET** | `net` | $R_n$ | net radiation | 净辐射 |
| **SZA** | `zenith` | $Z$ | solar zenith angle | 太阳天顶角 |
| **cosSZA** | `mu0` | $\mu_0$ | cosine of SZA | 余弦天顶角 |
| **SAA** | `azimuth` | $\phi$ | solar azimuth angle | 太阳方位角 |
| **GHIC** | `ghi_clear` | $G_{hc}$ | clear-sky GHI | 晴空水平总辐照度 |
| **BNIC** | `bni_clear` | $B_{nc}$ | clear-sky BNI | 晴空法向直接辐照度 |
| **DHIC** | `dhi_clear` | $D_{hc}$ | clear-sky DHI | 晴空水平散射辐照度 |
| **LWDC** | `lwd_clear` | $L_{dc}$ | clear-sky LWD | 晴空下行长波辐射 |
| **LWUC** | `lwu_clear` | $L_{uc}$ | clear-sky LWU | 晴空上行长波辐射 |
| **BNIE** | `bni_extra` | $E_{0n}$ | extraterrestrial BNI | 地外法向辐照度 |
| **GHIE** | `ghi_extra` | $E_{0}$ | extraterrestrial GHI | 地外水平辐照度 |
| **SC** | `solar_constant` | $E_{\text{sc}}$ | solar constant | 太阳常数 |
| **CSI** | `kappa` | $\kappa$ | clear-sky index | 晴空指数 |
| **k_t** | `kt` | $k_t$ | clearness index | 晴朗指数 |
| **K_t** (daily) | `Kt` | $K_t$ | daily clearness index | 日晴朗指数 |
| **k_b** | `kb` | $k_b$ | beam transmittance | 直射透射率 |
| **k_d** | `kd` | $k_d$ | diffuse transmittance | 散射透射率 |
| **k** | `k` | $k$ | diffuse fraction | 散射分数 |
| **TMP** | `temp` | $T$ | air temperature | 空气温度 |
| **RH** | `rh` | $RH$ | relative humidity | 相对湿度 |
| **SP** | `pressure` | $P$ | station pressure | 站点气压 |

---

## 🚩 Quality Control Flags
- For the data points that do not pass **PPL test**, we **MUST** use the following flag names:
    - `flagPPLGHI`: Flag for GHI physically possible limit test / GHI 物理可能范围测试标记
    - `flagPPLBNI`: Flag for BNI physically possible limit test / BNI 物理可能范围测试标记
    - `flagPPLDHI`: Flag for DHI physically possible limit test / DHI 物理可能范围测试标记
    - `flagPPLLWD`: Flag for LWD physically possible limit test / LWD 物理可能范围测试标记
- For the data points that do not pass **ERL test**, we **MUST** use the following flag names:
    - `flagERLGHI`: Flag for GHI extremely rare limit test / GHI 极罕见范围测试标记
    - `flagERLBNI`: Flag for BNI extremely rare limit test / BNI 极罕见范围测试标记
    - `flagERLDHI`: Flag for DHI extremely rare limit test / DHI 极罕见范围测试标记
    - `flagERLLWD`: Flag for LWD extremely rare limit test / LWD 极罕见范围测试标记
- For the data points that do not pass **Closure test**, we **MUST** use the following flag names:
    - `flag3lowSZA`: Flag for closure test at low SZA ($Z \le 75^\circ$) / 低太阳天顶角 ($\le 75^\circ$) 下的闭合测试标记
    - `flag3highSZA`: Flag for closure test at high SZA ($Z > 75^\circ$) / 高太阳天顶角 ($> 75^\circ$) 下的闭合测试标记
- For the data points that do not pass **Diffuse ratio (k) test**, we **MUST** use the following flag names:
    - `flagKKt`: Flag for combined $k$ and $k_t$ test ($k < 0.96$) / $k$ 和 $k_t$ 结合测试标记 ($k < 0.96$)
    - `flagKlowSZA`: Flag for diffuse ratio test at low SZA ($Z < 75^\circ$, $k < 1.05$) / 低太阳天顶角 ($Z < 75^\circ$) 下的散射分数测试标记 ($k < 1.05$)
    - `flagKhighSZA`: Flag for diffuse ratio test at high SZA ($Z \ge 75^\circ$, $k < 1.1$) / 高太阳天顶角 ($Z \ge 75^\circ$) 下的散射分数测试标记 ($k < 1.1$)
- For the data points that do not pass **k-index test**, we **MUST** use the following flag names:
    - `flagKbKt`: Flag for $k_b < k_t$ test / $k_b$ 小于 $k_t$ 测试标记
    - `flagKb`: Flag for $k_b$ physical limit test / $k_b$ 物理限值测试标记
    - `flagKt`: Flag for $k_t$ physical limit test / $k_t$ 物理限值测试标记
- For the data points that do not pass **Tracker-off test**, we **MUST** use the following flag name:
    - `flagTracker`: Flag for solar tracker failure detection / 太阳跟踪器失准检测标记

---

## 📝 Documentation & Coding Rules

### 1. Bilingual Comments
- **MUST** follow the length-dependent format:
    - If the combined line is shorter than 80 characters: Use `# English / 中文` format on a single line.
    - If the combined line exceeds 80 characters: Use two separate lines.
- Example (Short): `# Fetch data / 获取数据`
- Example (Long):
    ```python
    # Calculate extraterrestrial horizontal irradiance
    # 计算地外水平辐照度
    ```


### 2. Docstring Structure
- **MUST** use NumPy/SciPy style with bilingual descriptions.
- **MUST** include both English and Chinese in the summary and parameter descriptions.
- **MUST** include a `References` section at the end of the docstring for functions based on literature.

```python
def function_name(param):
    """
    English summary here.
    此处为中文摘要。

    Parameters
    ----------
    param : type
        English description.
        中文描述集。

    Returns
    -------
    result : type
        English description.
        中文描述。

    References
    ----------
    .. [1] Author, A. (Year). Title. Journal, Vol(Issue), Pages.
    """
```

### 3. Citations
- **MUST** follow **APA style** for all citations (e.g. Author, A. (Year). Title. Journal, Vol(Issue), Pages).
- **MUST** use the reStructuredText citation format (`.. [1]`) within the docstring's `References` section.
- **MUST** use sentence case for scientific paper titles in references.

### 4. Naming & Consistency
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
- **MUST** limit line length to a maximum of 110 characters.
- **MUST** use sentence case for scientific paper titles in references.

### 5. Function Signature Style
- **MUST** use compact signatures: required parameters first, then optional parameters with defaults.
- **MUST NOT** put the entire signature on one line if that line would exceed **110 characters** (same limit as §4). Prefer **one line** only when the full `def …` fits within 110 characters.
- If wrapping is needed, **MUST** split into **a few short lines** (typically **two** lines of parameters): group several parameters per line so **each line stays ≤ 110 characters**. **DO NOT** use an unnecessarily long single line; **DO NOT** default to one parameter per line unless the list is very long or each default is long.
- For continuation lines after a trailing comma, **MUST** indent so the wrapped parameters align with the style used in this repo (see `run_qc` in `bsrn/qc/wrapper.py` or `pretty_average` in `bsrn/utils/averaging.py`: **11 spaces** before the first token on the continuation line).
- Example (short, one line): `def add_clearsky_columns(df, station_code=None, lat=None, lon=None, elev=None, model="ineichen"):`
- Example (wrapped, two lines): `def plot_k_vs_kt(df, model, lat, lon, ghi_col="ghi", dhi_col="dhi", k_mod_col=None,\n                 min_ghi=50.0, output_file=None, title=None):`
- Example (short, package pattern): `def pretty_average(df, rule, alignment="floor", aggfunc="mean", resolution=None, match_ceiling_labels=True):`

### 6. Output Directory
- All generated plots and figures **MUST** be saved to the project root directory.
- **DO NOT** save output files into `tests/`, `src/`, or other source directories.

### 7. Git Usage
- **DO NOT** push to git unless explicitly instructed by the USER.
- **MUST** exclude all PDF files (`*.pdf`) from Git to avoid bloat in the repository.

### 8. BSRN File Handling Policy
- **Single-file workflow**: All high-level workflows (QC, clear-sky modeling, CSD, separation, visualization) **MUST** operate on **one BSRN station-to-archive file at a time** (for example, a single `XXXMMYY.dat.gz` monthly file).
- **No implicit concatenation**: Library functions **MUST NOT** silently concatenate multiple months or years internally. If users need multi-month analyses, they **MUST** loop over files and combine results explicitly at the application level.
- **Index scope**: Within a single run, functions **MUST** assume that the `DatetimeIndex` comes from a single contiguous BSRN monthly file; cross-file or cross-year assumptions are out of scope for the core package.

### 9. Visualization & Aesthetics
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
    - **9 pt** for **all** plotnine text (titles, axes, strip labels, legends, heatmap cell labels, etc.) across `src/bsrn/visualization/` unless a figure needs an explicit exception (document it in the module).
- **Format**: All plots **MUST** be output in **PDF** format to ensure high-quality vector graphics.
- **Color bar (continuous legend)**:
    - **MUST** match the setup used in `availability.py` for consistency across plots.
    - Use `scale_fill_cmap(cmap_name='viridis', name="<Legend title>")` for continuous fill scales.
    - In `theme()`: `legend_position="bottom"`, `legend_title=element_text(size=9)`, `legend_text=element_text(size=9)`, `legend_key_width=100`, `legend_key_height=5`, `legend_margin=-12`, `legend_box_spacing=0`. To add space between x-axis title and legend without shifting the legend, use `axis_title_x=element_text(size=9, margin={"b": 8})` (bottom margin in points).
    - **Plot margin** (optional): `plot_margin_top`, `plot_margin_right`, `plot_margin_bottom`, `plot_margin_left` (each in `[0, 1]`; use 0 for no extra margin).

