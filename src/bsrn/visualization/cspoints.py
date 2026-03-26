"""
Clear-sky detection (CSD) results: one page per day, four rows per page (one per CSD model).
晴空检测 (CSD) 结果：每天一页，每页四行（每行对应一种 CSD 模型）。
"""

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    geom_ribbon,
    facet_grid,
    theme_minimal,
    theme,
    element_text,
    labs,
    scale_x_datetime,
)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from bsrn.constants import BSRN_STATIONS, WONG_PALETTE
from bsrn.io.reader import read_station_to_archive
from bsrn.modeling.clear_sky import add_clearsky_columns
from bsrn.physics.geometry import get_solar_position, get_ghi_extra
from bsrn.utils import reno_csd, ineichen_csd, lefevre_csd, brightsun_csd

# Display order for the four CSD methods (rows top to bottom).
CSD_METHOD_ORDER = ("Reno", "Ineichen", "Lefevre", "BrightSun")


def plot_csd_booklet(file_path, output_file, station_code, df=None):
    """
    Generate a PDF: one page per day, four rows (CSD methods) x three columns (GHI, BNI, DHI).
    生成 PDF：每天一页，四行（CSD 方法）× 三列（GHI、BNI、DHI）。

    Clear-sky detection is driven by GHI (and each method's required inputs). The same
    is_clearsky flag is overlaid on the GHI, BNI, and DHI time series in each row.
    晴空检测以 GHI（及各方法所需输入）为输入；同一 is_clearsky 标记叠加到该行的 GHI、BNI、DHI 时间序列上。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (one month, e.g. .dat.gz).
        Ignored if `df` is provided.
        BSRN 站点存档文件路径（单月，如 .dat.gz）；提供 df 时忽略。
    output_file : str
        Path to the output PDF file.
        输出 PDF 路径。
    station_code : str
        Station abbreviation (e.g. "QIQ") for clear-sky and site location.
        站点缩写（如 "QIQ"），用于晴空与站点位置。
    df : pd.DataFrame, optional
        If provided, use this DataFrame instead of reading from file_path (for testing).
        Must have one-month DatetimeIndex and columns ghi, bni, dhi.
        若提供，则使用此 DataFrame 而非从文件读取（用于测试）；须为单月且含 ghi, bni, dhi。
    Returns
    -------
    None
        Saves the booklet to the given PDF path.
        将手册保存到指定 PDF。
    """

    # Load data and ensure December only.
    # 加载数据并确保仅包含十二月。
    if df is None:
        df = read_station_to_archive(file_path)
        if df is None:
            raise ValueError(f"Failed to read BSRN file: {file_path}")

    df = df.sort_index()
    unique_months = df.index.to_period("M").unique()
    if len(unique_months) != 1:
        raise ValueError(f"File must contain exactly one month. Found: {unique_months}")

    stn = BSRN_STATIONS.get(station_code, {"lat": 0, "lon": 0, "elev": 0})
    lat, lon, elev = stn["lat"], stn["lon"], stn["elev"]

    # Solar position and extraterrestrial GHI.
    # 太阳位置和地外水平辐照度。
    solpos = get_solar_position(df.index, lat, lon, elev)
    zenith = solpos["zenith"].values
    ghi_extra = np.asarray(get_ghi_extra(df.index, zenith), dtype=float)

    # Clear-sky columns (required for Reno, BrightSun, and for reference).
    # 晴空列（Reno、BrightSun 所需，以及参考）。
    df = add_clearsky_columns(df, station_code)

    ghi = df["ghi"].values
    bni = df["bni"].values
    dhi = df["dhi"].values
    ghi_clear = df["ghi_clear"].values
    bni_clear = df["bni_clear"].values
    dhi_clear = df["dhi_clear"].values
    times = df.index

    # Run all four CSD methods (inputs are GHI and method-specific: ghi_clear, ghi_extra, zenith, dhi, etc.).
    # 运行四种 CSD 方法（输入为 GHI 及各方法所需：ghi_clear、ghi_extra、zenith、dhi 等）。
    out_reno = reno_csd(ghi, ghi_clear, times=times)
    out_ineichen = ineichen_csd(ghi, ghi_extra, zenith, times=times)
    out_lefevre = lefevre_csd(ghi, dhi, ghi_extra, zenith, times=times)
    out_brightsun = brightsun_csd(zenith, ghi, ghi_clear, dhi, dhi_clear, times)

    # Build merged frame: time, ghi/bni/dhi, clear-sky cols, and one column per method.
    # 构建合并框架：时间、GHI/BNI/DHI、晴空列、每种方法一列。
    merged = pd.DataFrame(
        index=times,
        data={
            "ghi": ghi,
            "bni": bni,
            "dhi": dhi,
            "ghi_clear": ghi_clear,
            "bni_clear": bni_clear,
            "dhi_clear": dhi_clear,
            "is_reno": out_reno["is_clearsky"],
            "is_ineichen": out_ineichen["is_clearsky"],
            "is_lefevre": out_lefevre["is_clearsky"],
            "is_brightsun": out_brightsun["is_clearsky"],
        },
    )
    merged.index.name = "time"
    merged = merged.reset_index()

    # Long format: one row per (time, method), with ghi, bni, dhi and clear-sky, is_clearsky.
    # 长格式：每行对应 (time, method)，包含 ghi、bni、dhi 和晴空、is_clearsky。
    long_list = []
    for method in CSD_METHOD_ORDER:
        col = f"is_{method.lower()}"
        sub = merged[
            ["time", "ghi", "bni", "dhi", "ghi_clear", "bni_clear", "dhi_clear", col]
        ].copy()
        sub = sub.rename(columns={col: "is_clearsky"})
        sub["method"] = method
        long_list.append(sub)
    long_df = pd.concat(long_list, ignore_index=True)
    long_df["method"] = pd.Categorical(
        long_df["method"], categories=list(CSD_METHOD_ORDER), ordered=True
    )

    # Long format for plotting: GHI / BNI / DHI time series; overlay is_clearsky points on all three.
    # 绘图用长表：GHI / BNI / DHI 时间序列；在同一行的三个面板上叠加 is_clearsky 点。
    m1 = long_df.melt(
        id_vars=["time", "method", "is_clearsky"],
        value_vars=["ghi", "bni", "dhi"],
        var_name="param",
        value_name="measured",
    )
    m2 = long_df.melt(
        id_vars=["time", "method"],
        value_vars=["ghi_clear", "bni_clear", "dhi_clear"],
        var_name="param_clear",
        value_name="clearsky",
    )
    param_map = {"ghi": "GHI", "bni": "BNI", "dhi": "DHI"}
    m1["parameter"] = m1["param"].map(param_map)
    m2["parameter"] = m2["param_clear"].str.replace("_clear", "").map(param_map)
    plot_df = m1[["time", "method", "parameter", "measured", "is_clearsky"]].merge(
        m2[["time", "method", "parameter", "clearsky"]],
        on=["time", "method", "parameter"],
    )
    plot_df["parameter"] = pd.Categorical(
        plot_df["parameter"], categories=["GHI", "BNI", "DHI"], ordered=True
    )
    clear_only = plot_df.loc[plot_df["is_clearsky"] == True].copy()

    # Figure size: 160 mm width, height for 4 rows x 3 cols.
    # 图宽 160 mm，高为 4 行 × 3 列。
    width_inch = 160 / 25.4
    height_inch = width_inch / 3 * 4 / 1.4

    print(f"Generating CSD booklet: {output_file} ...")
    with PdfPages(output_file) as pdf:
        for date, day_df in plot_df.groupby(pd.to_datetime(plot_df["time"]).dt.date):
            day_clear = clear_only.loc[
                pd.to_datetime(clear_only["time"]).dt.date == date
            ]
            title = f"{station_code} - {date.strftime('%Y %b %d')} (CSD)"

            # Wong palette: [1] Sky Blue (clearsky/ribbon), [0] Orange (measured), [2] Bluish Green (CSD points)
            # Wong 配色方案：[1] 天蓝色（晴空/色带），[0] 橙色（实测），[2] 蓝绿色（CSD 点）
            ribbon_color = WONG_PALETTE[1]
            p = (
                ggplot(day_df, aes(x="time"))
                + geom_ribbon(
                    mapping=aes(ymin=0, ymax="clearsky"),
                    fill=ribbon_color,
                    alpha=0.25,
                )
                + geom_line(
                    aes(y="clearsky"), color=ribbon_color, size=0.2, alpha=0.7, linetype="dashed"
                )
                + geom_line(aes(y="measured"), color=WONG_PALETTE[0], size=0.5)
                + geom_point(
                    data=day_clear,
                    mapping=aes(y="measured"),
                    color=WONG_PALETTE[2],
                    size=0.3,
                    alpha=0.7,
                )
                + facet_grid("method ~ parameter", scales="free_y")
                + labs(title=title, x="Time (UTC)", y="[W/m²]")
                + theme_minimal()
                + theme(
                    text=element_text(family="Times New Roman", size=9),
                    axis_title=element_text(size=9),
                    axis_text=element_text(size=9),
                    plot_title=element_text(size=9, margin={"b": 10}),
                    figure_size=(width_inch, height_inch),
                    strip_text=element_text(size=9),
                )
                + scale_x_datetime(date_labels="%H:%M")
            )
            fig = p.draw()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("Done.")
