"""
CAMS solar radiation service (CRS) HTTP retrieval helpers.
CAMS 太阳辐射服务（CRS）HTTP 下载辅助函数。

CRS and McClear are both exposed on SoDa (same ``api.soda-solardata.com`` WPS endpoint);
this module mirrors :mod:`bsrn.io.mcclear` but uses WPS ``Identifier=get_cams_radiation``.
CRS 与 McClear 均在 SoDa 上提供（相同 ``api.soda-solardata.com`` WPS 端点）；
本模块与 :mod:`bsrn.io.mcclear` 平行，但使用 WPS ``Identifier=get_cams_radiation``。
"""

import io
import math
import pandas as pd
import requests
from bsrn.constants import (
    BSRN_STATIONS,
    CRS_API_HOST,
    CRS_HF_REPO_ID,
    CRS_HIMAWARI_EARTH_DISK_RADIUS_DEG,
    CRS_HIMAWARI_MIN_START_UTC,
    CRS_HIMAWARI_SUBSATELLITE_LAT_DEG,
    CRS_HIMAWARI_SUBSATELLITE_LON_DEG,
    CRS_INTEGRATED_COLUMNS,
    CRS_MAINTAINER_EMAIL,
    CRS_MSG_EARTH_DISK_RADIUS_DEG,
    CRS_MSG_MIN_START_UTC,
    CRS_MSG_SUBSATELLITE_LAT_DEG,
    CRS_MSG_SUBSATELLITE_LON_DEG,
    CRS_OUTPUT_COLUMNS,
    CRS_VARIABLE_MAP,
)


def _check_crs_coverage(latitude: float, longitude: float, start) -> None:
    """
    Require the site inside the Himawari or MSG **60° reliability disk** and *start* not
    before the applicable minimum (see :mod:`bsrn.constants`).
    要求站点落在 Himawari 或 MSG 的 **60° 可靠性圆盘**内，且 *start* 不早于对应最早日期。

    Parameters
    ----------
    latitude : float
        Site latitude. [degrees]
        站点纬度 [度]。
    longitude : float
        Site longitude. [degrees]
        站点经度 [度]。
    start : datetime-like
        Request period start (naive or tz-aware; compared in UTC calendar sense for min-date check).
        请求起始时间（无时区或带时区；与最早允许日期的比较按 UTC 日历语义）。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the site is outside both satellite disks or *start* is before the required minimum.
        站点不在任一盘内，或 *start* 早于要求的最小日期时。

    References
    ----------
    .. [1] CAMS radiation service — SoDa.
       https://www.soda-pro.com/web-services/radiation/cams-radiation-service
    """
    # Compare *start* as UTC-naive timestamp / 将起始时间规范为 UTC 无时区以便与常量日期比较
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is not None:
        start_cmp = start_ts.tz_convert("UTC").tz_localize(None)
    else:
        start_cmp = start_ts

    def _central_angle_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Great-circle central angle between two surface points. [degrees]
        两点间大圆中心角 [度]。
        """
        rlat1 = math.radians(lat1)
        rlat2 = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        # Haversine half-chord squared, clamped for numerical stability
        # Haversine 半弦平方，夹紧以保证数值稳定
        h = (
            math.sin(dlat / 2) ** 2
            + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
        )
        return 2 * math.degrees(math.asin(math.sqrt(min(1.0, max(0.0, h)))))

    def _in_disk(lat: float, lon: float, sslat: float, sslon: float, radius_deg: float) -> bool:
        """
        True if (lat, lon) is within *radius_deg* of subsatellite point.
        若 (lat, lon) 在星下点 *radius_deg* 内则为 True。
        """
        return _central_angle_deg(lat, lon, sslat, sslon) <= radius_deg

    # Himawari vs MSG footprint from constants / 由常量定义的两颗卫星覆盖圆盘
    in_himawari = _in_disk(
        latitude,
        longitude,
        CRS_HIMAWARI_SUBSATELLITE_LAT_DEG,
        CRS_HIMAWARI_SUBSATELLITE_LON_DEG,
        CRS_HIMAWARI_EARTH_DISK_RADIUS_DEG,
    )
    in_msg = _in_disk(
        latitude,
        longitude,
        CRS_MSG_SUBSATELLITE_LAT_DEG,
        CRS_MSG_SUBSATELLITE_LON_DEG,
        CRS_MSG_EARTH_DISK_RADIUS_DEG,
    )

    if not in_himawari and not in_msg:
        raise ValueError(
            "Site is outside the Himawari and MSG 60° reliability disks "
            "(see CRS_HIMAWARI_* and CRS_MSG_* in bsrn.constants). / "
            "站点不在 Himawari 与 MSG 的 60° 可靠性圆盘内（见 bsrn.constants）。"
        )

    # Earliest allowed *start*: union of applicable satellite minima / 各适用卫星最早日期的并集（取最大）
    min_start = pd.Timestamp(CRS_MSG_MIN_START_UTC)
    if in_himawari:
        min_start = max(min_start, pd.Timestamp(CRS_HIMAWARI_MIN_START_UTC))
    if in_msg:
        min_start = max(min_start, pd.Timestamp(CRS_MSG_MIN_START_UTC))

    if start_cmp < min_start:
        raise ValueError(
            f"CRS request start must be on or after {min_start.date()} for this location. / "
            f"该位置 CRS 请求起始日期应不早于 {min_start.date()}。"
        )


def _parse_crs(raw_or_buffer):
    """
    Parse SoDa CAMS CRS CSV into the project irradiance frame (used by ``download_crs`` only).
    将 SoDa CAMS CRS CSV 解析为项目辐照度 DataFrame（仅由 ``download_crs`` 使用）。

    Parameters
    ----------
    raw_or_buffer : str or file-like
        Raw SoDa CAMS text or readable text buffer.
        SoDa CAMS 原始文本或可读文本缓冲区。

    Returns
    -------
    data : pd.DataFrame
        UTC index and columns ``ghi_crs``, ``bni_crs``, ``dhi_crs`` [W/m²] only.
        UTC 索引与列 ``ghi_crs``、``bni_crs``、``dhi_crs`` [W/m²] 仅此。

    Raises
    ------
    ValueError
        Missing header line, missing columns after rename, or unreadable CSV.
        缺少表头行、重命名后缺列或 CSV 无法读取时。

    References
    ----------
    .. [1] CAMS radiation service — SoDa.
       https://www.soda-pro.com/web-services/radiation/cams-radiation-service
    """
    if isinstance(raw_or_buffer, str):
        fbuf = io.StringIO(raw_or_buffer)
    else:
        fbuf = raw_or_buffer

    # Skip preamble until column-name row / 跳过前言直至含列名的 “# Observation period” 行
    while True:
        line = fbuf.readline()
        if not line:
            raise ValueError("Invalid CRS payload: header not found. / 无法找到表头。")
        line = line.rstrip("\n")
        if line.startswith("# Observation period"):
            names = line.lstrip("# ").split(";")
            break

    data = pd.read_csv(fbuf, sep=";", comment="#", header=None, names=names)
    # Interval bounds from first column / 从首列解析观测时段起止
    obs_period = data["Observation period"].str.split("/")
    # Using the second part of the period (end-time) for ceiling-style labeling.
    # 使用时段的第二部分（结束时间）进行向上对齐（ceiling）风格的标记。
    data.index = pd.to_datetime(obs_period.str[1], utc=True)

    # SoDa integrated irradiance → mean irradiance over the step / 积分量转为步长内平均辐照度 [W/m²]
    integrated_cols = [c for c in CRS_INTEGRATED_COLUMNS if c in data.columns]
    time_delta = pd.to_datetime(obs_period.str[1]) - pd.to_datetime(obs_period.str[0])
    hours = time_delta.dt.total_seconds() / 3600.0
    data[integrated_cols] = data[integrated_cols].divide(hours.tolist(), axis="rows")

    data.index.name = None
    data = data.rename(columns=CRS_VARIABLE_MAP)
    missing = [c for c in CRS_OUTPUT_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "CRS payload missing required columns after rename: "
            f"{missing}. / 重命名后缺少列：{missing}。"
        )
    return data[CRS_OUTPUT_COLUMNS].copy()


def download_crs(latitude, longitude, start, end, email, elev=None, summarization="PT01H", timeout=30):
    """
    Download and parse CAMS Radiation Service (CRS) time series from SoDa.
    从 SoDa 下载并解析 CAMS 辐射服务 (CRS) 时间序列。

    CRS provides **all-sky** satellite-derived irradiances (not a clear-sky model like McClear).
    Requests use ``time_ref=UT`` and ``verbose=false`` (fixed; not configurable).
    Parsed frame contains only UTC index and all-sky ``ghi_crs``, ``bni_crs``, ``dhi_crs`` [W/m²]
    (other SoDa fields are dropped). Location and *start* are validated by ``_check_crs_coverage``.
    CRS 提供**全天空**卫星反演辐照度（不同于 McClear 类晴空模型）。请求固定为 ``time_ref=UT``、``verbose=false``（不可配置）。
    解析结果仅含 UTC 索引与全天空 ``ghi_crs``、``bni_crs``、``dhi_crs`` [W/m²]（其余 SoDa 列丢弃）。
    地理位置与 *start* 由 ``_check_crs_coverage`` 校验。

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees. [degrees]
        十进制度纬度 [度]。
    longitude : float
        Longitude in decimal degrees. [degrees]
        十进制度经度 [度]。
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
        Station elevation [m]. If None, use SoDa default terrain lookup (-999).
        站点海拔高度 [米]。若为 None 则使用 SoDa 默认地形查找 (-999)。
    summarization : str, default "PT01H"
        ISO-8601 duration for temporal aggregation (e.g., "PT01M", "PT15M", "PT01H").
        时间聚合的 ISO-8601 时长（如 "PT01M"、"PT15M"、"PT01H" 等）。
    timeout : int, default 30
        HTTP request timeout in seconds.
        HTTP 请求超时时间（秒）。

    Returns
    -------
    data : pd.DataFrame
        Columns ghi_crs, bni_crs, dhi_crs only; UTC DatetimeIndex.
        仅列 ghi_crs、bni_crs、dhi_crs；UTC DatetimeIndex。

    Raises
    ------
    requests.HTTPError
        SoDa returned a non-success HTTP status (often with ows:ExceptionText in the body).
        SoDa 返回非成功 HTTP 状态（响应体常含 ows:ExceptionText）。
    ValueError
        Coverage or start failed _check_crs_coverage, XML instead of CSV, parse error, or empty data.
        _check_crs_coverage 校验、XML 非 CSV、解析失败或无数据时。

    References
    ----------
    .. [1] Schroedter-Homscheidt, M., et al. (2016). CAMS radiation service.
       European Commission.
    """
    if elev is None:
        elev = -999

    _check_crs_coverage(latitude, longitude, start)

    # WPS date strings in UTC / WPS 用的 UTC 日期字符串
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is not None:
        start_str = start_ts.tz_convert("UTC").strftime("%Y-%m-%d")
    else:
        start_str = start_ts.strftime("%Y-%m-%d")
    if end_ts.tzinfo is not None:
        end_str = end_ts.tz_convert("UTC").strftime("%Y-%m-%d")
    else:
        end_str = end_ts.strftime("%Y-%m-%d")
    # Double-encode @ for query string / 对 @ 二次编码以符合 SoDa 查询串约定
    email_encoded = email.replace("@", "%2540")

    # SoDa Execute payload (DataInputs) / SoDa Execute 的 DataInputs 键值对
    data_inputs_dict = {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": elev,
        "date_begin": start_str,
        "date_end": end_str,
        "time_ref": "UT",
        "summarization": summarization,
        "username": email_encoded,
        "verbose": "false",
    }
    data_inputs = ";".join([f"{key}={value}" for key, value in data_inputs_dict.items()])
    params = {
        "Service": "WPS",
        "Request": "Execute",
        "Identifier": "get_cams_radiation",
        "version": "1.0.0",
        "RawDataOutput": "irradiation",
    }

    base_url = f"https://{CRS_API_HOST}/service/wps"

    try:
        res = requests.get(
            base_url + "?DataInputs=" + data_inputs,
            params=params,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise requests.Timeout(
            f"CRS request timed out for {base_url}: {exc}"
        ) from exc

    # Enrich HTTPError with OWS exception text when present / 若有 OWS 异常文本则并入 reason
    if not res.ok:
        text = res.text or ""
        if "ows:ExceptionText" in text:
            try:
                errors = text.split("ows:ExceptionText")[1][1:-2]
            except Exception:
                errors = text
            res.reason = f"{res.reason}: <{errors}>"
        res.raise_for_status()

    body_text = res.content.decode("utf-8")
    stripped = body_text.lstrip()
    # 200 OK can still be XML on some errors / 部分错误仍可能以 200 返回 XML
    if stripped.startswith("<?xml") or stripped.startswith("<ows:ExceptionReport"):
        raise ValueError(
            "SoDa CRS returned XML instead of CSV. / SoDa CRS 返回 XML 而非 CSV。"
        )

    data = _parse_crs(body_text)

    if len(data.index) == 0:
        raise ValueError(
            "SoDa CRS returned no data rows. / SoDa CRS 未返回数据行。"
        )
    return data


def _hf_fetch_to_memory(repo_id, filename):
    """
    Fetch a file from Hugging Face Hub directly to memory (bytes).
    从 Hugging Face Hub 直接获取文件到内存（字节）。

    Parameters
    ----------
    repo_id : str
        Hugging Face repository ID (e.g., "dazhiyang/bsrn-v1").
        Hugging Face 仓库 ID。
    filename : str
        Path within the repository (e.g., "qiq/qiq0624_crs.parquet").
        仓库内的文件路径。

    Returns
    -------
    content : bytes
        Raw file bytes.
        原始文件字节。

    Raises
    ------
    ImportError
        If huggingface_hub is not installed.
        未安装 huggingface_hub 时。
    FileNotFoundError
        On 404 or other fetch failure; message includes maintainer contact.
        404 或其他获取失败时；信息含维护者联系方式。
    """
    try:
        from huggingface_hub import hf_hub_url
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for CRS. Install with: pip install huggingface_hub"
        )

    print(f"Fetching CRS from Hugging Face: {filename}")
    try:
        url = hf_hub_url(
            repo_id=repo_id, filename=filename, repo_type="dataset"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise FileNotFoundError(
                f"{filename} is not yet on huggingface, please contact the maintainer "
                f"Dazhi Yang at {CRS_MAINTAINER_EMAIL} for update."
            ) from e
        raise FileNotFoundError(
            f"{filename} is not yet on huggingface, please contact the maintainer "
            f"Dazhi Yang at {CRS_MAINTAINER_EMAIL} for update."
        ) from e
    except Exception as e:
        raise FileNotFoundError(
            f"{filename} is not yet on huggingface, please contact the maintainer "
            f"Dazhi Yang at {CRS_MAINTAINER_EMAIL} for update."
        ) from e


def _fetch_crs_from_hf(station_code, index):
    """
    Fetch raw bytes from Hugging Face for the months required by the index.
    从 Hugging Face 获取索引所需月份的原始字节。

    Parameters
    ----------
    station_code : str
        BSRN station code (case-insensitive).
        BSRN 站点代码（大小写不敏感）。
    index : pd.DatetimeIndex
        Non-empty target index; months present determine which files are fetched.
        非空目标索引；出现的月份决定拉取哪些文件。

    Returns
    -------
    contents : list of bytes
        One element per month, in sorted order.
        每月一个元素，按日期排序。

    Raises
    ------
    ValueError
        If index is empty.
        index 为空时。
    """
    if index.empty:
        raise ValueError("index must not be empty. / index 不能为空。")
    stn = station_code.lower()

    # We use a shift to correctly handle ceiling-aligned labels at month boundaries.
    # Labels at the exact start of a month (e.g. 00:00:00) belong to the PREVIOUS month.
    # 使用偏移来正确处理月份边界处的向上对齐标签；整月起始点（如 00:00:00）属于前一个月。
    shifted_index = index.shift(-1, freq="s")
    unique_months = sorted(set(zip(shifted_index.year, shifted_index.month)))

    contents = []
    for year, month in unique_months:
        yy = str(year)[2:]
        mm = f"{month:02d}"
        # Monthly filename format: qiq0124_crs.parquet / 月度文件格式：qiq0124_crs.parquet
        filename = f"{stn}{mm}{yy}_crs.parquet"
        hf_filename = f"{stn}/{filename}"
        content = _hf_fetch_to_memory(CRS_HF_REPO_ID, hf_filename)
        contents.append(content)
    return contents


def _load_crs_parquet(path_or_bytes):
    """
    Load one CRS parquet into a UTC-indexed DataFrame.
    将单个 CRS parquet 加载为 UTC 索引的 DataFrame。

    Parameters
    ----------
    path_or_bytes : str, path-like, bytes, or file-like
        Path to the parquet file, or bytes content, or file-like object.
        parquet 文件路径、字节内容或类文件对象。

    Returns
    -------
    data : pd.DataFrame
        CRS data with columns ghi_crs, bni_crs, dhi_crs and UTC DatetimeIndex.
        含 ghi_crs、bni_crs、dhi_crs 列及 UTC DatetimeIndex 的 CRS 数据。
    """
    if isinstance(path_or_bytes, bytes):
        path_or_bytes = io.BytesIO(path_or_bytes)
    data = pd.read_parquet(path_or_bytes)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("CRS parquet must have DatetimeIndex. / CRS parquet 必须有 DatetimeIndex。")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    return data


def fetch_crs_hf(index, station_code):
    """
    Fetch CRS from Hugging Face and return inputs aligned to target index.
    从 Hugging Face 获取 CRS 并返回对齐到目标索引的输入。

    Parameters
    ----------
    index : pd.DatetimeIndex
        Target time index to align CRS outputs to.
        需要对齐的目标时间索引。
    station_code : str
        BSRN station code (e.g., "QIQ").
        BSRN 站点代码（如 "QIQ"）。

    Returns
    -------
    aligned : pd.DataFrame
        CRS inputs reindexed to `index` with columns ghi_crs, bni_crs, dhi_crs.
        重新索引到 `index` 的 CRS 输入，含 ghi_crs, bni_crs, dhi_crs 列。
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            "index must be a pandas DatetimeIndex. / index 必须是 pandas DatetimeIndex。"
        )
    if index.empty:
        raise ValueError("index must not be empty. / index 不能为空。")

    contents = _fetch_crs_from_hf(station_code, index)
    dfs = [_load_crs_parquet(c) for c in contents]
    raw = pd.concat(dfs).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    # By default, use direct reindexing.
    # To align with ceiling average, we often need +1 hour shift here
    # but I'm leaving it as-is for now based on user's 'revert'.
    aligned = raw.reindex(index)
    return aligned


def add_crs_columns(df, station_code=None, lat=None, lon=None, elev=None):
    """
    Adds CRS (CAMS Radiation Service) all-sky columns to a DataFrame.
    Fetches data from Hugging Face automatically.
    向 DataFrame 添加 CRS (CAMS 辐射服务) 全天空辐射列。自动从 Hugging Face 获取数据。

    Location can be given by BSRN station code or by explicit lat/lon/elev.
    位置可由 BSRN 站点代码指定，或由显式的 lat/lon/elev 指定。

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to which columns will be added. Index must be DatetimeIndex.
        要添加列的 DataFrame。索引必须是 DatetimeIndex。
    station_code : str, optional
        BSRN station abbreviation. [e.g. 'QIQ'] Used if lat/lon/elev not provided.
        BSRN 站点缩写。[例如 'QIQ']。未提供 lat/lon/elev 时使用。
    lat : float, optional
        Latitude. [degrees] Required for non-BSRN stations if station_code omitted.
        纬度。[度]。非 BSRN 站点且未提供 station_code 时必填。
    lon : float, optional
        Longitude. [degrees] Required for non-BSRN stations if station_code omitted.
        经度。[度]。非 BSRN 站点且未提供 station_code 时必填。
    elev : float, optional
        Elevation. [m] Required for non-BSRN stations if station_code omitted.
        海拔。[米]。非 BSRN 站点且未提供 station_code 时必填。

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with added crs columns.
        增加了 CRS 列的输入 DataFrame。

    Raises
    ------
    ValueError
        If ``df.index`` is not a :class:`~pandas.DatetimeIndex`. / 索引非 DatetimeIndex。
    ValueError
        If neither a valid station_code nor complete (lat, lon, elev) is provided.
        若既未提供有效 station_code 也未提供完整的 (lat, lon, elev)。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex.")

    # Resolve metadata: explicit lat/lon/elev or BSRN lookup
    if lat is not None and lon is not None and elev is not None:
        pass  # use provided coordinates
    elif station_code is not None and station_code in BSRN_STATIONS:
        meta = BSRN_STATIONS[station_code]
        lat, lon, elev = meta["lat"], meta["lon"], meta["elev"]
    elif station_code is not None:
        raise ValueError(
            f"Station '{station_code}' not found in BSRN registry. "
            "For non-BSRN stations, provide 'lat', 'lon', and 'elev' explicitly. / "
            f"在 BSRN 注册表中未找到站点 '{station_code}'。非 BSRN 站点请显式提供 lat、lon、elev。"
        )
    else:
        raise ValueError(
            "Insufficient metadata. Provide a valid BSRN 'station_code' or "
            "explicit 'lat', 'lon', and 'elev'. / "
            "元数据不足。请提供有效的 BSRN 站点代码或显式的 lat、lon、elev。"
        )

    if station_code is None:
        raise ValueError("fetch_crs_hf currently requires 'station_code' to fetch parquets from Hugging Face.")

    crs_data = fetch_crs_hf(df.index, station_code)
    for col in crs_data.columns:
        df[col] = crs_data[col]
    return df

