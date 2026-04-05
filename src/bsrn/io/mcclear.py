"""
CAMS McClear HTTP retrieval helpers.
CAMS McClear HTTP 下载辅助函数。
"""

import io
import pandas as pd
import requests
from bsrn.constants import MCCLEAR_INTEGRATED_COLUMNS, MCCLEAR_VARIABLE_MAP, MCCLEAR_API_HOST


def _parse_mcclear(raw_or_buffer):
    """
    Parse SoDa McClear CSV into the project DataFrame (used by ``_download_mcclear`` only).
    将 SoDa McClear CSV 解析为项目 DataFrame（仅由 ``_download_mcclear`` 使用）。

    Parameters
    ----------
    raw_or_buffer : str or file-like
        Raw CAMS text or readable text buffer.
        CAMS 原始文本或可读文本缓冲区。

    Returns
    -------
    data : pd.DataFrame
        Parsed time-series data with UTC index for sub-daily resolutions.
        解析后的时间序列数据；亚日尺度为 UTC 索引。

    Raises
    ------
    ValueError
        If the McClear header line is missing or the payload is invalid.
        缺少 McClear 表头或载荷无效时。

    References
    ----------
    .. [1] CAMS McClear service info. (n.d.). SoDa.
       http://www.soda-pro.com/web-services/radiation/cams-mcclear/info
    """
    if isinstance(raw_or_buffer, str):
        fbuf = io.StringIO(raw_or_buffer)
    else:
        fbuf = raw_or_buffer

    # Read metadata header lines until column names line / 读取元数据表头直到列名行
    while True:
        line = fbuf.readline()
        if not line:
            raise ValueError("Invalid McClear payload: header not found. / 无法找到表头。")
        line = line.rstrip("\n")
        if line.startswith("# Observation period"):
            names = line.lstrip("# ").split(";")
            break

    data = pd.read_csv(fbuf, sep=";", comment="#", header=None, names=names)
    
    # Interval bounds from first column / 从首列解析观测时段起止
    obs_period = data["Observation period"].str.split("/")
    # Using the first part of the period (start-time) for floor-style labeling. 
    # 使用时段的第一部分（起始时间）进行向下对齐（floor）风格的标记。
    data.index = pd.to_datetime(obs_period.str[0], utc=True)

    # Convert Wh/m^2 to W/m^2 using interval duration / 依据时间区间长度将 Wh/m^2 转换为 W/m^2
    integrated_cols = [c for c in MCCLEAR_INTEGRATED_COLUMNS if c in data.columns]
    time_delta = pd.to_datetime(obs_period.str[1]) - pd.to_datetime(obs_period.str[0])
    hours = time_delta.dt.total_seconds() / 3600.0
    data[integrated_cols] = data[integrated_cols].divide(hours.tolist(), axis="rows")

    data.index.name = None
    data = data.rename(columns=MCCLEAR_VARIABLE_MAP)
    return data


def _download_mcclear(latitude, longitude, start, end, email, elev=None,
                      timeout=30):
    """
    Download and parse CAMS McClear from SoDa (used by ``fetch_mcclear`` only).
    从 SoDa 下载并解析 CAMS McClear（仅由 ``fetch_mcclear`` 调用）。

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees. [degrees]
        十进制度纬度。[度]
    longitude : float
        Longitude in decimal degrees. [degrees]
        十进制度经度。[度]
    start : datetime.datetime or pandas.Timestamp
        Start date (inclusive) of requested period.
        请求时间段的起始日期（含）。
    end : datetime.datetime or pandas.Timestamp
        End date (inclusive) of requested period.
        请求时间段的结束日期（含）。
    email : str
        SoDa account email.
        SoDa 账户邮箱。
    elev : float, optional
        Station elevation. [m] If None, use SoDa default terrain lookup.
        站点海拔高度。[米] 若为 None 则使用 SoDa 默认地形高程。
    timeout : int, default 30
        HTTP request timeout in seconds.
        HTTP 请求超时时间（秒）。

    Returns
    -------
    data : pd.DataFrame
        Parsed McClear data.
        解析后的 McClear 数据。

    Raises
    ------
    ValueError
        If the request starts before 2004-01-01 or the response is not valid CSV.
        起始日期早于 2004-01-01，或响应非有效 CSV 时。
    requests.Timeout
        If the HTTP request exceeds *timeout*.
        HTTP 请求超过 *timeout* 时。
    requests.HTTPError
        If SoDa returns a non-success status after ``raise_for_status``.
        SoDa 返回非成功 HTTP 状态时。

    References
    ----------
    .. [1] Lefèvre, M., Oumbe, A., Blanc, P., Espinar, B., Gschwind, B., Qu, Z.,
       et al. (2013). McClear: A new model estimating downwelling solar
       radiation at ground level in clear-sky conditions. Atmospheric
       Measurement Techniques, 6(9), 2403–2418.
    .. [2] Gschwind, B., Wald, L., Blanc, P., Lefèvre, M., Schroedter-Homscheidt, M.,
       & Arola, A. (2019). Improving the McClear model estimating the downwelling
       solar radiation at ground level in cloud-free conditions – McClear-v3.
       Meteorologische Zeitschrift, 28(2).
    """
    if elev is None:
        elev = -999

    # McClear availability: service is defined from 2004-01-01 onward.
    # McClear 可用性：服务从 2004-01-01 起提供。
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is not None:
        start_cmp = start_ts.tz_convert("UTC").tz_localize(None)
    else:
        start_cmp = start_ts
    if start_cmp < pd.Timestamp("2004-01-01"):
        raise ValueError(
            "McClear data are only available from 2004-01-01 onward. / "
            "McClear 数据仅在 2004-01-01 之后可用。"
        )

    # Format dates and username for SoDa request / 为 SoDa 请求格式化日期和用户名
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_str = start_ts.tz_convert("UTC").strftime("%Y-%m-%d")
    else:
        start_str = start_ts.strftime("%Y-%m-%d")
    if end_ts.tzinfo is not None:
        end_str = end_ts.tz_convert("UTC").strftime("%Y-%m-%d")
    else:
        end_str = end_ts.strftime("%Y-%m-%d")
    email_encoded = email.replace("@", "%2540")

    # Build WPS DataInputs payload for McClear (1‑min, UT) / 构建 McClear 的 WPS DataInputs 载荷（1 分钟、UT）
    data_inputs_dict = {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": elev,
        "date_begin": start_str,
        "date_end": end_str,
        "time_ref": "UT",
        "summarization": "PT01M",
        "username": email_encoded,
        "verbose": "false",
    }
    data_inputs = ";".join([f"{key}={value}" for key, value in data_inputs_dict.items()])
    params = {
        "Service": "WPS",
        "Request": "Execute",
        "Identifier": "get_mcclear",
        "version": "1.0.0",
        "RawDataOutput": "irradiation",
    }

    # Use the same HTTPS endpoint and request pattern as pvlib.iotools.get_cams,
    # with the host defined in project constants.
    # 使用与 pvlib.iotools.get_cams 相同的 HTTPS 端点和请求格式，主机名由项目常量统一管理。
    base_url = f"https://{MCCLEAR_API_HOST}/service/wps"

    try:
        res = requests.get(
            base_url + "?DataInputs=" + data_inputs,
            params=params,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise requests.Timeout(
            f"McClear request timed out for {base_url}: {exc}"
        ) from exc

    # If an error occurs on the server side, CAMS returns a PyWPS-style XML/HTML
    # with ows:ExceptionText; bubble that up for easier debugging.
    # 服务器端出错时，CAMS 会返回包含 ows:ExceptionText 的 PyWPS 风格 XML/HTML，将其拼接到错误信息中便于调试。
    if not res.ok:
        text = res.text or ""
        if "ows:ExceptionText" in text:
            try:
                errors = text.split("ows:ExceptionText")[1][1:-2]
            except Exception:
                errors = text
            res.reason = f"{res.reason}: <{errors}>"
        res.raise_for_status()

    # Successful responses are CSV; parse directly from memory.
    # 成功响应为 CSV；直接在内存中解析。
    fbuf = io.StringIO(res.content.decode("utf-8"))
    data = _parse_mcclear(fbuf)
    return data


def fetch_mcclear(index, latitude, longitude, elev, email, timeout=30):
    """
    Retrieve and align McClear data to a target DatetimeIndex.
    获取并对齐 McClear 数据到给定的 DatetimeIndex。

    Parameters
    ----------
    index : pd.DatetimeIndex
        Target time index to align McClear outputs to.
        需要对齐的目标时间索引。
    latitude : float
        Latitude in decimal degrees. [degrees]
        十进制度纬度。[度]
    longitude : float
        Longitude in decimal degrees. [degrees]
        十进制度经度。[度]
    elev : float
        Site elevation. [m]
        站点海拔。[米]
    email : str
        SoDa account email.
        SoDa 账户邮箱。
    timeout : int, default 30
        HTTP request timeout in seconds.
        HTTP 请求超时时间（秒）。

    Returns
    -------
    aligned : pd.DataFrame
        McClear data reindexed to `index`. Must contain
        `ghi_clear`, `bni_clear`, and `dhi_clear`.
        重新索引到 `index` 的 McClear 数据，包含
        `ghi_clear`、`bni_clear` 与 `dhi_clear` 列。

    Raises
    ------
    ValueError
        If ``index`` is not a DatetimeIndex, McClear columns are missing, or the
        downloaded frame has an invalid index.
        ``index`` 非 DatetimeIndex、McClear 缺列或下载数据索引无效时。
    requests.Timeout
        Propagated from :func:`_download_mcclear` when the HTTP call times out.
        由 :func:`_download_mcclear` 在 HTTP 超时时向上传递。
    requests.HTTPError
        Propagated from SoDa on HTTP failure.
        SoDa HTTP 失败时向上传递。
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            "index must be a pandas DatetimeIndex. / index 必须是 pandas DatetimeIndex。"
        )

    # Determine inclusive date range from index / 从索引确定包含的起止日期
    start = pd.Timestamp(index.min()).to_pydatetime()
    end = pd.Timestamp(index.max()).to_pydatetime()

    data = _download_mcclear(
        latitude=latitude,
        longitude=longitude,
        start=start,
        end=end,
        email=email,
        elev=elev,
        timeout=timeout,
    )

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "McClear data index must be DatetimeIndex. / McClear 数据索引必须为 DatetimeIndex。"
        )

    data = data.copy()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    required_cols = {"ghi_clear", "bni_clear", "dhi_clear"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(
            f"McClear data missing required columns: {sorted(missing)}"
        )

    aligned = data.reindex(index)
    return aligned

