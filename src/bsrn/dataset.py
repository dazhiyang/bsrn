"""
Central BSRN dataset: one monthly station file as a typed, validated object.

Encapsulates station identity, resolved geographic metadata, and minute-
resolution data in a single Pydantic model. ``lr0100`` is the source of
truth for minute data; ``data()`` returns a cached ``DataFrame`` with
only the mean/value columns by default. LR0300 / LR4000 columns are
available on demand via ``data(include=[...])``. Most pipeline methods
(solar position, clear-sky, QC) mutate the cached frame in-place;
the ``average`` method replaces the cache with a coarser-index result
from :func:`~bsrn.utils.averaging.pretty_average`.

BSRN 中心数据集：将一个月度站点文件封装为带类型校验的对象。``lr0100``
为分钟数据源；``data()`` 返回仅含均值列的缓存 ``DataFrame``。
LR0300 / LR4000 列按需通过 ``data(include=[...])`` 获取。多数管线方法
原地修改缓存；``average`` 方法以
:func:`~bsrn.utils.averaging.pretty_average` 的较粗索引结果替换缓存。
"""

from __future__ import annotations

import calendar
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import (BaseModel, ConfigDict, PrivateAttr,
                      field_validator, model_validator)

from .archive.records_models import LR0100, LR0300, LR4000
from .constants import BSRN_STATIONS
from .io.reader import read_bsrn_archive

# Variable maps: LR field name → short column name exposed by data().
# Only these columns appear; _std/_min/_max are dropped.
# 变量映射：LR 字段名 → data() 暴露的短列名。
_LR0100_VAR_MAP = {
    "ghi_avg": "ghi", "bni_avg": "bni",
    "dhi_avg": "dhi", "lwd_avg": "lwd",
    "temperature": "temp", "humidity": "rh",
    "pressure": "pressure",
}
_LR0300_VAR_MAP = {
    "swu_avg": "swu", "lwu_avg": "lwu", "net_avg": "net",
}
_LR4000_VAR_MAP = {
    "domeT1_down": "dt1d", "domeT2_down": "dt2d",
    "domeT3_down": "dt3d", "bodyT_down": "btd",
    "domeT1_up": "dt1u", "domeT2_up": "dt2u",
    "domeT3_up": "dt3u", "bodyT_up": "btu",
}


class BSRNDataset(BaseModel):
    """
    One monthly BSRN dataset: station identity + minute data.

    单月 BSRN 数据集：站点标识 + 分钟级数据。

    Typical enrichment on the cached :meth:`data` frame is
    :meth:`solpos`, then :meth:`clear_sky`, then :meth:`qc`
    (each mutates that frame in place and returns it). :meth:`average`
    replaces the cache with a coarser time series from
    :func:`~bsrn.utils.averaging.pretty_average`.

    常见流程为在缓存的 :meth:`data` 帧上依次调用
    :meth:`solpos`、:meth:`clear_sky`、:meth:`qc`
    （均原地修改该帧并返回同一对象）。:meth:`average` 以
    :func:`~bsrn.utils.averaging.pretty_average` 的结果替换缓存（较粗时间索引）。

    Parameters
    ----------
    station_code : str
        Three-letter BSRN station code (must exist in
        ``BSRN_STATIONS``).
        三位 BSRN 站点代码（须在 ``BSRN_STATIONS`` 中）。
    year : int
        Four-digit measurement year.
        四位测量年份。
    month : int
        Measurement month (1--12).
        测量月份（1--12）。
    lr0100 : LR0100
        Validated LR0100 logical record (minute radiation and
        met data). This is the source of truth; ``data()`` is
        derived from it.
        已校验的 LR0100 逻辑记录（分钟辐射与气象数据）。
        为数据源；``data()`` 由其派生。
    lr0300 : LR0300 or None
        Validated LR0300 logical record (reflected / upward
        SW, LW, net radiation). Default ``None``.
        已校验的 LR0300 逻辑记录（反射/上行辐射）。
    lr4000 : LR4000 or None
        Validated LR4000 logical record (pyrgeometer minute
        data). Default ``None``.
        已校验的 LR4000 逻辑记录（长波表分钟数据）。
    station_name : str or None
        Resolved from ``BSRN_STATIONS`` when omitted.
        省略时从 ``BSRN_STATIONS`` 解析。
    lat : float or None
        Latitude (degrees); resolved from ``BSRN_STATIONS``.
        纬度（度）；从 ``BSRN_STATIONS`` 解析。
    lon : float or None
        Longitude (degrees); resolved from ``BSRN_STATIONS``.
        经度（度）；从 ``BSRN_STATIONS`` 解析。
    elev : float or None
        Elevation (m above sea level); resolved from
        ``BSRN_STATIONS``.
        海拔（米）；从 ``BSRN_STATIONS`` 解析。
    resolution : int or None
        Temporal resolution in minutes (e.g. ``1``, ``2``,
        ``3``, ``5``). Defaults to ``1``.
        时间分辨率（分钟）；默认 ``1``。

    Raises
    ------
    ValueError
        If ``station_code`` is not in ``BSRN_STATIONS``,
        ``month`` is out of 1--12.
        ``station_code`` 不在 ``BSRN_STATIONS`` 或 ``month``
        不在 1--12 时。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # ------------------------------------------------------------------ #
    #  Fields                                                              #
    # ------------------------------------------------------------------ #

    # Core (required): identity + minute radiation data.
    # 核心（必填）：站点标识 + 分钟辐射数据。
    station_code: str
    year: int
    month: int
    lr0100: LR0100

    # Optional logical records. / 可选逻辑记录。
    lr0300: Optional[LR0300] = None
    lr4000: Optional[LR4000] = None

    # Resolved from BSRN_STATIONS; users may override.
    # 从 BSRN_STATIONS 解析；用户可覆盖。
    station_name: str = None
    lat: float = None
    lon: float = None
    elev: float = None
    resolution: int = None

    # Internal cached DataFrame; excluded from serialisation.
    _df_cache: Optional[pd.DataFrame] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    #  Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("station_code")
    @classmethod
    def _validate_station_code(cls, v):
        code = v.upper()
        if code not in BSRN_STATIONS:
            raise ValueError(
                f"unknown station code {v!r}; "
                f"not in BSRN_STATIONS"
            )
        return code

    @field_validator("month")
    @classmethod
    def _validate_month(cls, v):
        if v < 1 or v > 12:
            raise ValueError(
                f"month must be 1--12, got {v}"
            )
        return v

    @model_validator(mode="after")
    def _resolve_metadata(self):
        """
        Fill ``station_name``, ``lat``, ``lon``, ``elev`` from
        ``BSRN_STATIONS`` when not explicitly provided.

        未显式传入时从 ``BSRN_STATIONS`` 填充站点名、经纬度、
        海拔。
        """
        meta = BSRN_STATIONS[self.station_code]
        if self.station_name is None:
            self.station_name = meta["name"]
        if self.lat is None:
            self.lat = meta["lat"]
        if self.lon is None:
            self.lon = meta["lon"]
        if self.elev is None:
            self.elev = meta["elev"]
        if self.resolution is None:
            self.resolution = self._infer_resolution()
        return self

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(cls, path):
        """
        Parse a BSRN ``.dat.gz`` station-to-archive file and return
        a fully validated ``BSRNDataset``.

        解析 BSRN ``.dat.gz`` 台站存档文件并返回完整校验的
        ``BSRNDataset``。

        Parameters
        ----------
        path : str or Path
            Path to the ``.dat.gz`` file (filename format
            ``XXXMMYY.dat.gz``).
            ``.dat.gz`` 文件路径（文件名格式
            ``XXXMMYY.dat.gz``）。

        Returns
        -------
        BSRNDataset

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the filename cannot be parsed or no LR0100 block
            is found.
        """
        return cls(**read_bsrn_archive(path))

    # ------------------------------------------------------------------ #
    #  data()                                                              #
    # ------------------------------------------------------------------ #

    def data(self, include=None):
        """
        Minute-resolution DataFrame derived from ``lr0100``.

        由 ``lr0100`` 派生的分钟分辨率 DataFrame。

        The base frame contains only the LR0100 **mean / scalar**
        columns under short names (``ghi``, ``bni``, ``dhi``,
        ``lwd``, ``temp``, ``rh``, ``pressure``). It is built once
        and cached so that pipeline methods (``solpos``, ``average``, etc.)
        can enrich it in-place.

        基础帧仅包含 LR0100 均值/标量列的短名。首次构建后缓存，
        管线方法可原地扩展。

        Parameters
        ----------
        include : sequence of str, optional
            Extra logical records to merge: ``"lr0300"`` and/or
            ``"lr4000"`` (case-insensitive).  When given, the
            corresponding mean/value columns are appended.
            要合并的额外逻辑记录（不区分大小写）。

        Returns
        -------
        pandas.DataFrame
            UTC ``DatetimeIndex``; default columns are LR0100
            means only.
        """
        if self._df_cache is None:
            self._df_cache = self._build_base_frame()

        if not include:
            return self._df_cache

        want = {s.lower() for s in include}
        extra = {}

        if "lr0300" in want and self.lr0300 is not None:
            for lr_col, short in _LR0300_VAR_MAP.items():
                vec = getattr(self.lr0300, lr_col, None)
                if vec is not None:
                    extra[short] = np.asarray(vec)

        if "lr4000" in want and self.lr4000 is not None:
            for lr_col, short in _LR4000_VAR_MAP.items():
                vec = getattr(self.lr4000, lr_col, None)
                if vec is not None:
                    extra[short] = np.asarray(vec)

        if not extra:
            return self._df_cache

        return self._df_cache.assign(**extra)

    @property
    def plot(self):
        """
        Accessor for built-in plotting routines.

        内置绘图程序的适配器。
        """
        return BSRNPlot(self)

    # ------------------------------------------------------------------ #
    #  Pipeline methods (delegate to standalone functions)                  #
    #  管线方法（委托给独立函数）                                           #
    # ------------------------------------------------------------------ #

    def solpos(self):
        """
        Add solar position and extraterrestrial irradiance columns
        to the cached ``data()`` frame.

        向缓存的 ``data()`` 帧添加太阳位置和地外辐射列。

        Delegates to
        :func:`~bsrn.physics.geometry.add_solpos_columns` using
        the resolved ``lat``, ``lon``, ``elev``.

        Returns
        -------
        pandas.DataFrame
            ``data()`` with added columns: ``zenith``,
            ``apparent_zenith``, ``azimuth``, ``bni_extra``,
            ``ghi_extra``.
        """
        from .physics.geometry import add_solpos_columns
        return add_solpos_columns(
            self.data(), station_code=self.station_code,
            lat=self.lat, lon=self.lon, elev=self.elev,
        )

    def clear_sky(self, model="ineichen", mcclear_email=None):
        """
        Add clear-sky irradiance columns to the cached ``data()``
        frame.

        向缓存的 ``data()`` 帧添加晴空辐射列。

        Delegates to
        :func:`~bsrn.modeling.clear_sky.add_clearsky_columns`.

        Parameters
        ----------
        model : str, optional
            Clear-sky model name (default ``'ineichen'``).
            晴空模型名称（默认 ``'ineichen'``）。
        mcclear_email : str, optional
            E-mail for CAMS McClear API (only when
            ``model='mcclear'``).
            CAMS McClear API 邮箱（仅 ``model='mcclear'``
            时使用）。

        Returns
        -------
        pandas.DataFrame
            ``data()`` with added clear-sky columns
            (``ghi_clear``, ``bni_clear``, ``dhi_clear``, …).
        """
        from .modeling.clear_sky import add_clearsky_columns
        return add_clearsky_columns(
            self.data(), station_code=self.station_code,
            lat=self.lat, lon=self.lon, elev=self.elev,
            model=model, mcclear_email=mcclear_email,
        )

    def qc(self, tests=('ppl', 'erl', 'closure',
                        'diff_ratio', 'k_index', 'tracker')):
        """
        Run QC tests and add flag columns to the cached ``data()``
        frame.

        运行 QC 测试并向缓存的 ``data()`` 帧添加标志列。

        Delegates to :func:`~bsrn.qc.wrapper.run_qc`.

        Parameters
        ----------
        tests : tuple of str, optional
            QC test names to run (default: all six).
            要运行的 QC 测试名称（默认全部六项）。

        Returns
        -------
        pandas.DataFrame
            ``data()`` with added ``flag_*`` columns.
        """
        from .qc.wrapper import run_qc
        return run_qc(
            self.data(), station_code=self.station_code,
            lat=self.lat, lon=self.lon, elev=self.elev,
            tests=tests,
        )

    def average(self, freq, alignment="ceiling", aggfunc="mean",
                match_ceiling_labels=True):
        """
        Time-average the cached ``data()`` with explicit labeled windows.

        使用显式标签窗对缓存的 ``data()`` 做时间平均。

        Delegates to :func:`~bsrn.utils.averaging.pretty_average` and
        **replaces** the internal cache with the returned frame (new index).

        委托 :func:`~bsrn.utils.averaging.pretty_average`，并以返回帧**替换**
        内部缓存（新索引）。

        Native timestep for **center** windows is taken from ``self.resolution``
        (minutes) when set; otherwise passed as ``None`` for
        :func:`~bsrn.utils.averaging.pretty_average` to infer.

        **center** 窗的原生步长取自 ``self.resolution``（分钟）；未设置时传
        ``None`` 由 :func:`~bsrn.utils.averaging.pretty_average` 推断。

        Parameters
        ----------
        freq : str
            Fixed bin frequency (e.g. ``'1h'``, ``'30min'``).
            固定分箱频率。
        alignment : {'floor', 'ceiling', 'center'}, optional
            Window alignment (default ``'ceiling'``).
            窗对齐方式（默认 ``'ceiling'``）。
        aggfunc : str or callable, optional
            Aggregation function (default ``'mean'``).
            聚合函数（默认 ``'mean'``）。
        match_ceiling_labels : bool, optional
            When ``alignment='center'``, monthly edge trim style (default
            ``True``, ceiling-like).
            ``alignment='center'`` 时的月界裁剪方式（默认 ``True``，类 ceiling）。

        Returns
        -------
        pandas.DataFrame
            One row per output label; also stored as the new cache.

        Raises
        ------
        TypeError
            If ``data().index`` is not a :class:`~pandas.DatetimeIndex`.
            ``data()`` 索引非 :class:`~pandas.DatetimeIndex` 时。
        ValueError
            Propagated from :func:`~bsrn.utils.averaging.pretty_average` when
            ``freq`` is not a fixed frequency.
            ``freq`` 非固定频率等由 ``pretty_average`` 抛出。
        """
        from .utils.averaging import pretty_average
        res = None
        if self.resolution is not None:
            res = pd.Timedelta(minutes=int(self.resolution))
        out = pretty_average(
            self.data(), freq, alignment=alignment, aggfunc=aggfunc,
            resolution=res, match_ceiling_labels=match_ceiling_labels,
        )
        self._df_cache = out
        return out

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_base_frame(self):
        """Build the LR0100-means-only DataFrame (called once)."""
        y, m = map(int, self.lr0100.yearMonth.split("-"))
        n = None
        cols = {}
        for lr_col, short in _LR0100_VAR_MAP.items():
            vec = getattr(self.lr0100, lr_col, None)
            if vec is not None:
                arr = np.asarray(vec)
                cols[short] = arr
                if n is None:
                    n = len(arr)
        if n is None:
            n = calendar.monthrange(y, m)[1] * 1440
        idx = pd.date_range(
            f"{y}-{m:02d}-01", periods=n,
            freq=f"{self.resolution} min", tz="UTC",
        )
        return pd.DataFrame(cols, index=idx)

    def _infer_resolution(self):
        """
        Infer temporal resolution from ``lr0100`` vector length
        vs calendar month.

        根据 ``lr0100`` 向量长度与日历月推断时间分辨率。
        """
        y, m = map(int, self.lr0100.yearMonth.split("-"))
        ndays = calendar.monthrange(y, m)[1]
        expected_1min = ndays * 1440
        for lr_col in _LR0100_VAR_MAP:
            vec = getattr(self.lr0100, lr_col, None)
            if vec is not None:
                n = len(vec)
                return expected_1min // n
        return 1

# ---------------------------------------------------------------------- #
#  BSRNPlot Accessor Class
# ---------------------------------------------------------------------- #

class BSRNPlot:
    """
    Visualization accessor for BSRNDataset.
    
    BSRNDataset 的可视化适配器。
    """

    def __init__(self, ds: "BSRNDataset"):
        self._ds = ds

    def __call__(self, dates, output_file=None, **kwargs):
        """
        Default to daily time series plot.
        默认为画日时间序列图。
        """
        return self.daily(dates, output_file=output_file, **kwargs)

    def daily(self, dates, output_file=None, **kwargs):
        """
        Plot daily daily plots (automatically delegates to day or booklet mode).
        画时间序列图（根据输入的日期自动生成单日图或多页手册图）。

        Parameters
        ----------
        dates : str, pd.Timestamp, or sequence
            Date or dates to plot.
            绘图日期或日期序列。
        output_file : str
            Output path for the plot.
            输出图像的路径。

        Returns
        -------
        None
        """
        import numpy as np
        import pandas as pd
        from .visualization.daily import (
            plot_bsrn_daily_day,
            plot_bsrn_daily_booklet,
        )

        df = self._ds.data()

        is_multi = isinstance(dates, (list, tuple, pd.Index, np.ndarray)) and len(dates) > 1

        if is_multi:
            # Filter the dataframe to only include the requested dates
            date_objs = pd.to_datetime(dates).date
            mask = np.isin(df.index.date, date_objs)
            filtered_df = df.loc[mask].copy()

            return plot_bsrn_daily_booklet(
                file_path=None,
                output_file=output_file,
                station_code=self._ds.station_code,
                df=filtered_df,
                **kwargs,
            )
        else:
            # Single date given
            date_to_plot = dates[0] if isinstance(dates, (list, tuple, pd.Index, np.ndarray)) else dates
            return plot_bsrn_daily_day(
                file_path=None,
                day=date_to_plot,
                station_code=self._ds.station_code,
                output_file=output_file,
                df=df,
                **kwargs,
            )

    def table(self, output_file=None, title=None):
        """
        Plot the QC results summary table.
        画质量控制结果汇总表。

        Parameters
        ----------
        output_file : str, optional
            Output path for the plot.
            输出图像的路径。
        title : str, optional
            Plot title.
            图表标题。

        Returns
        -------
        None
        """
        from .visualization.table import plot_table
        from .utils.quality import get_daily_stats

        daily_stats = get_daily_stats(
            self._ds.data(),
            self._ds.lat,
            self._ds.lon,
            self._ds.elev,
            station_code=self._ds.station_code,
        )

        return plot_table(daily_stats, title=title, output_file=output_file)


