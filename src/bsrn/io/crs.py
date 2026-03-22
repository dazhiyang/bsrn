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
import re
import pandas as pd
import requests
from huggingface_hub import hf_hub_url
from bsrn.constants import (
    BSRN_STATIONS,
    CRS_API_HOST,
    CRS_HF_REPO_ID,
    CRS_INTEGRATED_COLUMNS,
    HF_MAINTAINER_EMAIL,
    CRS_OUTPUT_COLUMNS,
    CRS_VARIABLE_MAP,
    CRS_HIMAWARI_MIN_START_UTC,
    CRS_MSG_MIN_START_UTC,
)
from bsrn.physics.geometry import in_satellite_disk
from bsrn.io.retrieval import get_bsrn_file_inventory, months_from_ftp_filenames

# ---------------------------------------------------------------------------
#  Private helpers / 内部辅助函数
# ---------------------------------------------------------------------------

def _crs_min_start_utc(latitude, longitude):
    """
    Get the earliest allowed CRS start date [UTC] for a given site.
    获取给定站点最早允许的 CRS 起始日期 [UTC]。

    Parameters
    ----------
    latitude : float
        Site latitude. [degrees]
    longitude : float
        Site longitude. [degrees]

    Returns
    -------
    min_start : pd.Timestamp or None
        The earliest start date, or None if the site is outside all disks.
        最早起始日期，若不在圆盘范围内则返回 None。
    """
    in_himawari = in_satellite_disk(latitude, longitude, "Himawari")
    in_msg = in_satellite_disk(latitude, longitude, "MSG")

    if not in_himawari and not in_msg:
        return None

    # Earliest allowed start: union of applicable satellite minima (favor earliest)
    # 适用卫星最早日期的并集（优先取较早者，即各适用的最小值中最小）
    candidates = []
    if in_himawari:
        candidates.append(pd.Timestamp(CRS_HIMAWARI_MIN_START_UTC))
    if in_msg:
        candidates.append(pd.Timestamp(CRS_MSG_MIN_START_UTC))

    return min(candidates)


def _check_crs_coverage(latitude, longitude, start):
    """
    Require the site inside the Himawari or MSG **60° reliability disk** and *start* not
    before the applicable minimum.
    要求站点落在 Himawari 或 MSG 的 **60° 可靠性圆盘**内，且 *start* 不早于对应最早日期。

    Parameters
    ----------
    latitude : float
        Site latitude. [degrees]
    longitude : float
        Site longitude. [degrees]
    start : datetime-like
        Request period start.

    Raises
    ------
    ValueError
        If the site is outside both satellite disks or *start* is before the required minimum.
    """
    min_start = _crs_min_start_utc(latitude, longitude)
    if min_start is None:
        raise ValueError(
            "Site is outside the Himawari (140.7°E) and MSG (0°E) 60° reliability disks. / "
            "站点不在 Himawari 与 MSG 的 60° 可靠性圆盘内。"
        )

    # Compare *start* as UTC-naive timestamp / 将起始时间规范为 UTC 无时区以便与常量日期比较
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is not None:
        start_cmp = start_ts.tz_convert("UTC").tz_localize(None)
    else:
        start_cmp = start_ts

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

    # Skip preamble until column-name row / 跳过前言直至含列名的 "# Observation period" 行
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
    # Using the first part of the period (start-time) for floor-style labeling.
    # 使用时段的第一部分（起始时间）进行向下对齐（floor）风格的标记。
    data.index = pd.to_datetime(obs_period.str[0], utc=True)

    # SoDa integrated irradiance → mean irradiance over the step / 积分量转为步长内平均辐照度 [W/m²]
    integrated_cols = [c for c in CRS_INTEGRATED_COLUMNS if c in data.columns]
    time_delta = pd.to_datetime(obs_period.str[1]) - pd.to_datetime(obs_period.str[0])
    hours = time_delta.dt.total_seconds() / 3600.0
    data[integrated_cols] = data[integrated_cols].divide(hours.tolist(), axis="rows")

    data.index.name = None # Remove index name / 移除索引名
    data = data.rename(columns=CRS_VARIABLE_MAP)
    missing = [c for c in CRS_OUTPUT_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "CRS payload missing required columns after rename: "
            f"{missing}. / 重命名后缺少列：{missing}。"
        )
    return data[CRS_OUTPUT_COLUMNS].copy()


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
    """
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
                f"Dazhi Yang at {HF_MAINTAINER_EMAIL} for update."
            ) from e
        raise FileNotFoundError(
            f"{filename} is not yet on huggingface, please contact the maintainer "
            f"Dazhi Yang at {HF_MAINTAINER_EMAIL} for update."
        ) from e
    except Exception as e:
        raise FileNotFoundError(
            f"{filename} is not yet on huggingface, please contact the maintainer "
            f"Dazhi Yang at {HF_MAINTAINER_EMAIL} for update."
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

    # With floor-labeling (start-of-interval), the month in the index directly 
    # determines which monthly file we need to fetch.
    # 使用向下对齐（floor，时段开始）标记后，索引中的月份直接决定了需要获取的月度文件。
    unique_months = sorted(set(zip(index.year, index.month)))

    contents = [] # Accumulate raw bytes / 累积原始字节
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
        data.index = data.index.tz_localize("UTC") # Localize to UTC / 规范化为 UTC 时区
    else:
        data.index = data.index.tz_convert("UTC") # Convert to UTC / 转换为 UTC 时区
    return data


# ---------------------------------------------------------------------------
#  Public API / 公开接口
# ---------------------------------------------------------------------------

def check_crs_availability(stations, username, password):
    """
    Check which BSRN stations are geographically covered by CAMS Radiation Service (CRS)
    **and** have BSRN archive files overlapping the CRS temporal range.
    检查哪些 BSRN 站点在 CAMS 辐射服务 (CRS) 的地理覆盖范围内，**且**其 BSRN 存档文件
    与 CRS 的年份范围存在交集。

    Workflow:
    1. Filter *stations* by spatial coverage (MSG and Himawari disks).
    2. Query BSRN FTP for the covered subset to obtain file inventories.
    3. Extract years from filenames and intersect with the CRS year range.

    Parameters
    ----------
    stations : list of str
        BSRN station codes to check (e.g. ``['BIL', 'BON', 'DRA']``).
        要检查的 BSRN 站点代码。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。

    Returns
    -------
    availability : dict
        A dictionary mapping station codes to availability metadata:
        ``{station_code: {'years': [list of years], 'months': [list of (y,m) tuples]}}``.
        ``years`` is used for bulk API downloads, and ``months`` for monthly 
        parquet writing. Stations with no overlap are omitted.
        ``{站点代码: {'years': [年份列表], 'months': [(年, 月) 元组列表]}}``。
        ``years`` 用于批量下载，``months`` 用于生成月度 parquet。无交集站点被省略。
    """
    # Mission start years for MSG and Himawari / MSG 与 Himawari 的任务起始年份
    y_min_msg = 2004
    y_min_hima = 2016
    y_max = pd.Timestamp.now(tz="UTC").year

    # Step 1: geographic filter / 地理覆盖过滤
    covered = {}  # maps station to its min_year
    for code in stations:
        code_upper = code.upper()
        if code_upper not in BSRN_STATIONS:
            continue
        meta = BSRN_STATIONS[code_upper]
        lat, lon = meta["lat"], meta["lon"]

        # Use library logic to determine coverage and minimum start date
        # 使用统一逻辑确定覆盖范围及最早日期
        in_msg = in_satellite_disk(lat, lon, "MSG")
        in_hima = in_satellite_disk(lat, lon, "Himawari")

        if in_msg or in_hima:
            # Union of applicable satellite minima / 适用卫星最早日期的并集
            min_y = y_min_msg
            if in_hima and not in_msg:
                min_y = y_min_hima
            elif in_hima and in_msg:
                min_y = min(y_min_msg, y_min_hima)
            covered[code_upper] = min_y

    if not covered:
        return {}

    # Step 2: FTP inventory for covered stations / 查询覆盖站点的 FTP 文件清单
    inventory = get_bsrn_file_inventory(list(covered.keys()), username, password)

    # Step 3: extract years and intersect with CRS range / 提取年份并与 CRS 范围取交集
    availability = {}
    for stn, files in inventory.items():
        stn_upper = stn.upper()
        min_y = covered[stn_upper]
        
        # Standardize month extraction / 标准化月份提取
        all_months = months_from_ftp_filenames(files)
        ym_filtered = [(y, m) for y, m in all_months if min_y <= y <= y_max]

        if ym_filtered:
            unique_years = sorted(list(set(y for y, m in ym_filtered)))
            # Store metadata for station / 存储站点的元数据
            availability[stn_upper] = {
                "years": unique_years,
                "months": sorted_ym
            }

    return availability


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
    .. [1] Schroedter-Homscheidt, M., et al. (2016). User's Guide to the CAMS Radiation Service.
       European Commission.
    """
    if elev is None:
        elev = -999 # Default to terrain lookup / 默认采用地形查找

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
