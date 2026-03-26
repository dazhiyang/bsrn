"""
Clear-sky model comparison booklet: one page per day, three panels (GHI, BNI, DHI).
晴空模型对比手册：每天一页，三个面板（GHI、BNI、DHI）。
"""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_line,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_datetime,
    theme,
    theme_minimal,
)

from bsrn.constants import WONG_PALETTE
from bsrn.io.reader import read_station_to_archive
from bsrn.modeling.clear_sky import add_clearsky_columns


def plot_clearsky_models_booklet(
    file_path,
    output_file,
    station_code,
    mcclear_email=None,
    df=None,
):  # noqa: PLR0913
    """
    Generate a monthly PDF booklet comparing measured irradiance with clear-sky models [1]_ [2]_ [3]_.
    生成月度 PDF 手册，对比实测辐照度与晴空模型结果。

    One page is created per day. Each page has three panels for GHI, BNI, and DHI.
    Measured series is black solid; Ineichen, McClear, and REST2 are Wong colors and dashed.
    每天生成一页。每页三个面板：GHI、BNI、DHI。
    实测为黑色实线；Ineichen、McClear、REST2 使用 Wong 配色并为虚线。

    Parameters
    ----------
    file_path : str
        Path to one BSRN station-to-archive monthly file (.dat.gz).
        单个 BSRN 站点月文件路径（.dat.gz）。
    output_file : str
        Output PDF path.
        输出 PDF 路径。
    station_code : str
        BSRN station abbreviation, used by clear-sky model calls.
        BSRN 站点缩写，用于晴空模型调用。
    mcclear_email : str, optional
        Email required by McClear API.
        McClear API 所需邮箱。
    df : pd.DataFrame, optional
        If provided, use this DataFrame instead of reading from file_path.
        若提供，则使用该 DataFrame 而非读取 file_path。

    Returns
    -------
    None
        Save booklet to `output_file`.
        将手册保存到 `output_file`。

    References
    ----------
    .. [1] Ineichen, P., & Perez, R. (2002). A new airmass independent formulation for the
       Linke turbidity coefficient. Solar Energy, 73(3), 151-157.
    .. [2] Lefèvre, M., Oumbe, A., Blanc, P., Espinar, B., Gschwind, B., Qu, Z., et al. (2013).
       McClear: A new model estimating downwelling solar radiation at ground level in clear-sky
       conditions. Atmospheric Measurement Techniques, 6(9), 2403-2418.
    .. [3] Gueymard, C. A. (2008). REST2: High-performance solar radiation model for cloudless-sky
       irradiance, illuminance, and photosynthetically active radiation: Validation with a
       benchmark dataset. Solar Energy, 82(3), 272-285.
    """
    if df is None:
        df = read_station_to_archive(file_path)
        if df is None:
            raise ValueError(f"Failed to read BSRN file: {file_path}")

    df = df.sort_index()
    unique_months = df.index.to_period("M").unique()
    if len(unique_months) != 1:
        raise ValueError(f"Input must contain exactly one month. Found: {unique_months}")

    required = {"ghi", "bni", "dhi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    # Keep only measured core variables for model calls.
    # 仅保留模型计算所需的实测核心变量。
    base = df[["ghi", "bni", "dhi"]].copy()

    # Run clear-sky models. McClear optional when mcclear_email is None.
    # 运行晴空模型。mcclear_email 为 None 时 McClear 可选。
    out_ineichen = add_clearsky_columns(base.copy(), station_code=station_code, model="ineichen")
    if mcclear_email:
        out_mcclear = add_clearsky_columns(
            base.copy(),
            station_code=station_code,
            model="mcclear",
            mcclear_email=mcclear_email,
        )
    else:
        out_mcclear = None
    try:
        out_rest2 = add_clearsky_columns(
            base.copy(), station_code=station_code, model="rest2"
        )
    except FileNotFoundError:
        out_rest2 = None

    # Build model list and style maps / 构建模型列表与样式映射
    models = ["Measured", "Ineichen"]
    if out_mcclear is not None:
        models.append("McClear")
    if out_rest2 is not None:
        models.append("REST2")
    width_inch = 160 / 25.4
    height_inch = width_inch / 3 * 1 / 1.4
    param_order = ["GHI", "BNI", "DHI"]
    model_order = models
    color_map = {
        "Measured": "#000000",
        "Ineichen": WONG_PALETTE[0],
        "McClear": WONG_PALETTE[1],
        "REST2": WONG_PALETTE[2],
    }
    linetype_map = {
        "Measured": "solid",
        "Ineichen": "dashed",
        "McClear": "dashed",
        "REST2": "dashed",
    }

    print(f"Generating clear-sky comparison booklet: {output_file} ...")
    with PdfPages(output_file) as pdf:
        for date, day_base in base.groupby(base.index.date):
            day_ineichen = out_ineichen.loc[day_base.index]
            plot_dict = {
                "time": day_base.index,
                "ghi_measured": day_base["ghi"].to_numpy(dtype=float),
                "bni_measured": day_base["bni"].to_numpy(dtype=float),
                "dhi_measured": day_base["dhi"].to_numpy(dtype=float),
                "ghi_ineichen": day_ineichen["ghi_clear"].to_numpy(dtype=float),
                "bni_ineichen": day_ineichen["bni_clear"].to_numpy(dtype=float),
                "dhi_ineichen": day_ineichen["dhi_clear"].to_numpy(dtype=float),
            }
            ghi_vars = ["ghi_measured", "ghi_ineichen"]
            bni_vars = ["bni_measured", "bni_ineichen"]
            dhi_vars = ["dhi_measured", "dhi_ineichen"]
            if out_mcclear is not None:
                day_mcclear = out_mcclear.loc[day_base.index]
                plot_dict["ghi_mcclear"] = day_mcclear["ghi_clear"].to_numpy(dtype=float)
                plot_dict["bni_mcclear"] = day_mcclear["bni_clear"].to_numpy(dtype=float)
                plot_dict["dhi_mcclear"] = day_mcclear["dhi_clear"].to_numpy(dtype=float)
                ghi_vars.append("ghi_mcclear")
                bni_vars.append("bni_mcclear")
                dhi_vars.append("dhi_mcclear")
            if out_rest2 is not None:
                day_rest2 = out_rest2.loc[day_base.index]
                plot_dict["ghi_rest2"] = day_rest2["ghi_clear"].to_numpy(dtype=float)
                plot_dict["bni_rest2"] = day_rest2["bni_clear"].to_numpy(dtype=float)
                plot_dict["dhi_rest2"] = day_rest2["dhi_clear"].to_numpy(dtype=float)
                ghi_vars.append("ghi_rest2")
                bni_vars.append("bni_rest2")
                dhi_vars.append("dhi_rest2")
            value_vars = ghi_vars + bni_vars + dhi_vars
            day_plot = pd.DataFrame(plot_dict)
            long_df = day_plot.melt(
                id_vars=["time"],
                value_vars=value_vars,
                var_name="series",
                value_name="value",
            )
            split = long_df["series"].str.split("_", n=1, expand=True)
            long_df["parameter"] = split[0].str.upper()
            long_df["model"] = split[1].replace(
                {"measured": "Measured", "ineichen": "Ineichen", "mcclear": "McClear", "rest2": "REST2"}
            )
            long_df["parameter"] = pd.Categorical(long_df["parameter"], categories=param_order, ordered=True)
            long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)

            title = f"{station_code} - {pd.Timestamp(date).strftime('%Y %b %d')}"
            used_colors = {m: color_map[m] for m in model_order if m in color_map}
            used_linetypes = {m: linetype_map[m] for m in model_order if m in linetype_map}
            p = (
                ggplot(long_df, aes(x="time", y="value", color="model", linetype="model"))
                + geom_line(size=0.3)
                + facet_wrap("~parameter", nrow=1, ncol=3, scales="free_y")
                + scale_color_manual(values=used_colors)
                + scale_linetype_manual(values=used_linetypes)
                + scale_x_datetime(date_labels="%H:%M")
                + labs(title=title, x="Time (UTC)", y="[W/m²]", color="", linetype="")
                + theme_minimal()
                + theme(
                    text=element_text(family="Times New Roman", size=9),
                    axis_title=element_text(size=9),
                    axis_text=element_text(size=9),
                    plot_title=element_text(size=9),
                    strip_text=element_text(size=9),
                    figure_size=(width_inch, height_inch),
                    legend_position="top",
                )
            )
            fig = p.draw()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Done.")
