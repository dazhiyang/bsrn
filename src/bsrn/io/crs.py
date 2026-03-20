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
    CRS_API_HOST,
    CRS_HIMAWARI_EARTH_DISK_RADIUS_DEG,
    CRS_HIMAWARI_MIN_START_UTC,
    CRS_HIMAWARI_SUBSATELLITE_LAT_DEG,
    CRS_HIMAWARI_SUBSATELLITE_LON_DEG,
    CRS_INTEGRATED_COLUMNS,
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
        Site latitude [degrees].
        站点纬度 [度]。
    longitude : float
        Site longitude [degrees].
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
        Great-circle central angle between two surface points [degrees].
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
    data.index = pd.to_datetime(obs_period.str[0], utc=True)

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
    从 SoDa 下载并解析 CAMS 辐射服务（CRS）时间序列。

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
        Station elevation. [m] If None, use SoDa default terrain lookup (-999).
        站点海拔高度。[米] 若为 None 则使用 SoDa 默认地形查找（-999）。
    summarization : str, default ``\"PT01H\"``
        ISO-8601 duration for temporal aggregation (e.g. ``\"PT01M\"``, ``\"PT15M\"``,
        ``\"PT01H\"``, ``\"P01D\"``, ``\"P01M\"``).
        时间聚合的 ISO-8601 时长（如 ``\"PT01M\"``、``\"PT15M\"``、``\"PT01H\"`` 等）。
    timeout : int, default 30
        HTTP request timeout in seconds.
        HTTP 请求超时时间（秒）。

    Returns
    -------
    data : pd.DataFrame
        Columns ``ghi_crs``, ``bni_crs``, ``dhi_crs`` only; UTC :class:`~pandas.DatetimeIndex`.
        仅列 ``ghi_crs``、``bni_crs``、``dhi_crs``；UTC :class:`~pandas.DatetimeIndex`。

    Raises
    ------
    requests.HTTPError
        SoDa returned a non-success HTTP status (often with ``ows:ExceptionText`` in the body).
        SoDa 返回非成功 HTTP 状态（响应体常含 ``ows:ExceptionText``）。
    ValueError
        Coverage or *start* failed :func:`_check_crs_coverage`, XML instead of CSV, parse error,
        or empty data.
        :func:`_check_crs_coverage`、XML 非 CSV、解析失败或无数据时。
    requests.Timeout
        Request exceeded *timeout*.

    References
    ----------
    .. [1] CAMS radiation service — SoDa.
       https://www.soda-pro.com/web-services/radiation/cams-radiation-service
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
