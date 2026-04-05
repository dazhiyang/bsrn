"""
NSRDB solar radiation database retrieval helpers.
NSRDB 太阳辐射数据库下载辅助函数。

Supports three variants (conus, full-disc, aggregated) via NLR API and Hugging Face.
通过 NLR API 与 Hugging Face 支持三种变体（conus、full-disc、aggregated）。
"""

import io
import math
import re
import pandas as pd
import requests
from huggingface_hub import hf_hub_url

from bsrn.constants import (
    BSRN_STATIONS,
    NSRDB_API_BASE_URL,
    HF_MAINTAINER_EMAIL,
    NSRDB_OUTPUT_COLUMNS,
    NSRDB_VARIABLE_MAP,
    NSRDB_VARIANTS,
)
from bsrn.physics.geometry import in_satellite_disk
from bsrn.io.retrieval import get_bsrn_file_inventory, months_from_ftp_filenames


# ---------------------------------------------------------------------------
#  Private helpers / 内部辅助函数
# ---------------------------------------------------------------------------



def _in_nsrdb_coverage(lat, lon, variant):
    """
    Check if (lat, lon) is within the spatial coverage of an NSRDB variant.
    检查 (lat, lon) 是否在 NSRDB 变体的空间覆盖范围内。

    Applies bounding-box filter (if defined) then satellite disk geometry.
    先应用边界框过滤（如已定义），再检查卫星圆盘几何。

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees. [degrees]
        十进制度纬度 [度]。
    lon : float
        Longitude in decimal degrees. [degrees]
        十进制度经度 [度]。
    variant : str
        NSRDB variant name: 'conus', 'full-disc', or 'aggregated'.
        NSRDB 变体名称。

    Returns
    -------
    bool
        True if location is within the variant's spatial footprint.
        若位置在变体空间覆盖范围内则为 True。
    """
    v = NSRDB_VARIANTS[variant]
    bbox = v.get("bbox")
    if bbox is not None:
        lat_lo, lat_hi = bbox["lat"]
        lon_lo, lon_hi = bbox["lon"]
        if not (lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi):
            return False
    return any(in_satellite_disk(lat, lon, sk) for sk in v["satellites"])


def _parse_nsrdb(raw_text):
    """
    Parse NSRDB API CSV payload (skips metadata row, handles units).
    解析 NSRDB API CSV 响应（跳过元数据行，处理单位）。

    Parameters
    ----------
    raw_text : str
        CSV response body from the NLR API.
        NLR API 返回的 CSV 响应正文。

    Returns
    -------
    data : pd.DataFrame
        UTC :class:`~pandas.DatetimeIndex` and project-standard irradiance columns.
        UTC DatetimeIndex 及项目标准辐照度列。
    """
    # NSRDB CSV has metadata on line 0, header on line 1, data starts on line 2
    # NSRDB CSV 第 0 行为元数据，第 1 行为表头，第 2 行起为数据
    df = pd.read_csv(io.StringIO(raw_text), skiprows=2)

    # Convert to UTC DatetimeIndex
    # 转换为 UTC DatetimeIndex
    df["dt"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]], utc=True)
    df = df.set_index("dt")

    # Rename and select columns
    # 重命名并选择列
    df = df.rename(columns=NSRDB_VARIABLE_MAP)
    valid_cols = [c for c in NSRDB_OUTPUT_COLUMNS if c in df.columns]
    return df[valid_cols].copy()


def _hf_fetch_to_memory(repo_id, filename):
    """
    Fetch a file from Hugging Face Hub directly to memory (bytes).
    从 Hugging Face Hub 直接获取文件到内存（字节）。

    Parameters
    ----------
    repo_id : str
        Hugging Face repository ID.
    filename : str
        Path within the repository.

    Returns
    -------
    content : bytes
        Raw file bytes.
        原始文件字节。

    Raises
    ------
    FileNotFoundError
        If the dataset file is missing (HTTP 404) on the Hub.
        Hub 上数据集文件缺失（HTTP 404）时。
    requests.HTTPError
        On other non-success HTTP statuses.
        其他非成功 HTTP 状态时。
    """
    print(f"Fetching NSRDB from Hugging Face: {filename}")
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise FileNotFoundError(
                f"{filename} not on HF Hub. Contact {HF_MAINTAINER_EMAIL} for updates."
            ) from e
        raise


def _fetch_nsrdb_from_hf(station_code, index, variant="conus"):
    """
    Fetch monthly NSRDB parquet blobs from Hugging Face for months in *index*.
    按 *index* 所需月份从 Hugging Face 拉取月度 NSRDB parquet 字节块。

    Parameters
    ----------
    station_code : str
        BSRN station code (case-insensitive).
        BSRN 站点代码（大小写不敏感）。
    index : pd.DatetimeIndex
        Non-empty target index; shifted month boundaries select files.
        非空目标索引；经边界平移后的月份决定文件。
    variant : str, default ``"conus"``
        NSRDB variant key in :data:`~bsrn.constants.NSRDB_VARIANTS`.
        :data:`~bsrn.constants.NSRDB_VARIANTS` 中的 NSRDB 变体键。

    Returns
    -------
    contents : list of bytes
        Raw parquet bytes per successfully fetched month (missing months skipped).
        每个成功获取的月份对应原始字节（缺失月份跳过）。
    """
    if index.empty:
        raise ValueError("index must not be empty.")
    stn = station_code.lower()
    v = NSRDB_VARIANTS[variant]
    v_folder = v["folder"]
    hf_repo_id = v["hf_repo_id"]

    # Align to months. Using -1s shift for boundary labels.
    shifted = index.shift(-1, freq="s")
    unique_months = sorted(set(zip(shifted.year, shifted.month)))

    contents = []
    for year, month in unique_months:
        yy = str(year)[2:]
        mm = f"{month:02d}"
        # Filename updated to match the new convention with variant suffix
        # 文件名更新为包含变体后缀的新约定
        filename = f"{stn}{mm}{yy}_nsrdb_{variant}.parquet"
        # HF repo structure is {stn}/{filename}
        hf_filename = f"{stn}/{filename}"
        try:
            content = _hf_fetch_to_memory(hf_repo_id, hf_filename)
            contents.append(content)
        except (FileNotFoundError, requests.HTTPError):
            # If a month is missing, we continue and reindexing will fill NaNs
            continue
    return contents


def _load_nsrdb_parquet(path_or_bytes, target_index=None):
    """
    Load one NSRDB parquet and optionally interpolate to *target_index*.
    加载单个 NSRDB parquet，并可选择插值到 *target_index*。

    Parameters
    ----------
    path_or_bytes : str, path-like, bytes, or file-like
        Parquet path, bytes, or readable buffer.
        parquet 路径、字节或可读缓冲区。
    target_index : pd.DatetimeIndex or None
        If given, reindex and time-interpolate to this index.
        若给定则重索引并按时间插值到该索引。

    Returns
    -------
    data : pd.DataFrame
        UTC-indexed NSRDB columns (possibly interpolated).
        UTC 索引的 NSRDB 列（可能已插值）。
    """
    if isinstance(path_or_bytes, bytes):
        path_or_bytes = io.BytesIO(path_or_bytes)
    data = pd.read_parquet(path_or_bytes)

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    if target_index is not None:
        # Reindex and interpolate to 1-min or other target resolution
        # 重索引并插值到 1 分钟或其他目标分辨率
        data = data.reindex(target_index.union(data.index)).sort_index()
        data = data.interpolate(method="time").reindex(target_index)

    return data


# ---------------------------------------------------------------------------
#  Public API / 公开接口
# ---------------------------------------------------------------------------

def check_nsrdb_availability(stations, username, password, variant="conus"):
    """
    Check which BSRN stations are geographically covered by an NSRDB variant
    **and** have BSRN archive files overlapping the variant's year range.
    检查哪些 BSRN 站点在 NSRDB 变体的地理覆盖范围内，**且**其 BSRN 存档文件
    与变体的年份范围存在交集。

    Workflow:
    1. Filter *stations* by spatial coverage (bbox + satellite disk).
    2. Query BSRN FTP for the covered subset to obtain file inventories.
    3. Extract years from filenames and intersect with the variant's year range.

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
    variant : str, default "conus"
        NSRDB variant name: 'conus', 'full-disc', or 'aggregated'.
        NSRDB 变体名称。

    Returns
    -------
    availability : dict
        A dictionary mapping station codes to availability metadata:
        ``{station_code: {'years': [list of years], 'months': [list of (y,m) tuples]}}``.
        ``years`` is used for bulk API downloads, and ``months`` for monthly 
        parquet writing. Stations with no overlap are omitted.
        ``{站点代码: {'years': [年份列表], 'months': [(年, 月) 元组列表]}}``。
        ``years`` 用于批量下载，``months`` 用于生成月度 parquet。无交集站点被省略。

    Raises
    ------
    ValueError
        If *variant* is not a recognised NSRDB variant name.
        *variant* 不是已知的 NSRDB 变体名称时。
    """

    if variant not in NSRDB_VARIANTS:
        raise ValueError(
            f"Unknown NSRDB variant: {variant}. / 未知的 NSRDB 变体。"
        )

    v = NSRDB_VARIANTS[variant]
    y_min, y_max = v["years"]

    # Step 1: geographic filter / 地理覆盖过滤
    covered = []
    for code in stations:
        code_upper = code.upper()
        if code_upper not in BSRN_STATIONS:
            continue
        meta = BSRN_STATIONS[code_upper]
        if _in_nsrdb_coverage(meta["lat"], meta["lon"], variant):
            covered.append(code_upper)

    if not covered:
        return {}

    # Step 2: FTP inventory for covered stations / 查询覆盖站点的 FTP 文件清单
    inventory = get_bsrn_file_inventory(covered, username, password)

    # Step 3: extract years and intersect with variant range / 提取年份并与变体范围取交集
    # BSRN filenames: e.g. pay0123.dat.gz or qiq0224.004
    # Pattern includes station code (3), month (2), year (2)
    availability = {}
    for stn, files in inventory.items():
        stn_upper = stn.upper()
        
        # Standardize month extraction / 标准化月份提取
        all_months = months_from_ftp_filenames(files)
        ym_filtered = [(y, m) for y, m in all_months if y_min <= y <= y_max]

        if ym_filtered:
            unique_years = sorted(list(set(y for y, m in ym_filtered)))
            availability[stn_upper] = {
                "years": unique_years,
                "months": sorted(list(set(ym_filtered)))  # Ensure unique and sorted
            }

    return availability


def download_nsrdb(latitude, longitude, year, api_key, email,
                   variant="conus", timeout=120):
    """
    Download NSRDB data from NLR API.
    从 NLR API 下载 NSRDB 数据。

    Parameters
    ----------
    latitude : float
        Site latitude [degrees].
        站点纬度 [度]。
    longitude : float
        Site longitude [degrees].
        站点经度 [度]。
    year : int
        Calendar year to download.
        要下载的公历年份。
    api_key : str
        NLR developer API key.
        NLR 开发者 API 密钥。
    email : str
        User email registered with the API.
        在 API 注册的用户邮箱。
    variant : str, default ``"conus"``
        NSRDB variant name (see :data:`~bsrn.constants.NSRDB_VARIANTS`).
        NSRDB 变体名称（见 :data:`~bsrn.constants.NSRDB_VARIANTS`）。
    timeout : int, default 120
        HTTP request timeout [s].
        HTTP 请求超时 [秒]。

    Returns
    -------
    df : pd.DataFrame
        NSRDB irradiance columns for the requested *year*.
        所请求 *year* 的 NSRDB 辐照度列。

    Raises
    ------
    ValueError
        If *year* is outside the variant's year range. / *year* 超出变体的年份范围时。
    ValueError
        If the location is not within the variant's spatial coverage.
        位置不在变体空间覆盖范围内时。

    References
    ----------
    .. [1] Sengupta, M., Xie, Y., Lopez, A., Habte, A., Maclaurin, G., & Shelby, J. (2018). The
           national solar radiation data base (NSRDB). Renewable and Sustainable Energy
           Reviews, 89, 51-60.
    .. [2] Xie, Y., Yang, J., Sengupta, M., Liu, Y., & Zhou, X. (2022). Improving the
           prediction of DNI with physics-based representation of all-sky circumsolar
           radiation. Solar Energy, 231, 758-766.
    .. [3] Xie, Y., Sengupta, M., Yang, J., Buster, G., Benton, B., Habte, A., & Liu, Y. (2023).
           Integration of a physics-based direct normal irradiance (DNI) model to enhance
           the National Solar Radiation Database (NSRDB). Solar energy, 266, 112195.
    .. [4] Xie, Y., Sengupta, M., & Dudhia, J. (2016). A Fast All-sky Radiation Model for
           Solar applications (FARMS): Algorithm and performance evaluation. Solar Energy,
           135, 435-445.
    """
    v = NSRDB_VARIANTS[variant]
    y_min, y_max = v["years"]
    if year is not None and not (y_min <= year <= y_max):
        raise ValueError(
            f"Year {year} outside range {y_min}–{y_max} for variant '{variant}'."
        )
    if not _in_nsrdb_coverage(latitude, longitude, variant):
        raise ValueError(f"Location not covered by variant '{variant}'.")

    url = f"{NSRDB_API_BASE_URL}{v['endpoint']}"

    params = {
        "api_key": api_key,
        "wkt": f"POINT({longitude} {latitude})",
        "attributes": "ghi,dni,dhi",
        "names": str(year),
        "utc": "true",
        "leap_day": "true",
        "interval": str(v["interval"]),
        "email": email,
        "affiliation": "BSRN Research",
        "reason": "academic research",
    }

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()

    # NREL-style errors sometimes return 200 OK with error msg in text
    if "error" in resp.text.lower() and len(resp.text) < 500:
        raise ValueError(f"NSRDB API Error: {resp.text}")

    return _parse_nsrdb(resp.text)


def fetch_nsrdb_hf(index, station_code, variant="conus"):
    """
    Fetch NSRDB from Hugging Face aligned to target index.
    从 Hugging Face 获取 NSRDB 并对齐到目标索引。

    Parameters
    ----------
    index : pd.DatetimeIndex
        Target grid (typically 1-minute BSRN timestamps).
        目标时间网格（通常为 BSRN 一分钟时间戳）。
    station_code : str
        BSRN station code for parquet naming on the Hub.
        Hub 上 parquet 命名所用的 BSRN 站点代码。
    variant : str, default ``"conus"``
        NSRDB variant (folder and filename suffix).
        NSRDB 变体（目录与文件名后缀）。

    Returns
    -------
    aligned : pd.DataFrame
        Columns from :data:`~bsrn.constants.NSRDB_OUTPUT_COLUMNS`, reindexed to *index*.
        来自 :data:`~bsrn.constants.NSRDB_OUTPUT_COLUMNS` 的列，重索引到 *index*。
    """
    contents = _fetch_nsrdb_from_hf(station_code, index, variant)
    if not contents:
        # Return empty frame with correct columns
        return pd.DataFrame(index=index, columns=NSRDB_OUTPUT_COLUMNS)

    dfs = [_load_nsrdb_parquet(c, target_index=index) for c in contents]
    aligned = pd.concat(dfs).sort_index()
    # Handle overlaps if any
    aligned = aligned[~aligned.index.duplicated(keep="first")]
    return aligned.reindex(index)


def add_nsrdb_columns(df, station_code=None, lat=None,
                      lon=None, elev=None, variant="conus"):
    """
    Adds NSRDB all-sky columns to a DataFrame.
    Fetches data from Hugging Face automatically.
    向 DataFrame 添加 NSRDB 全天空辐射列。自动从 Hugging Face 获取数据。

    Location can be given by BSRN station code or by explicit lat/lon/elev.
    位置可由 BSRN 站点代码指定，或由显式的 lat/lon/elev 指定。

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to which columns will be added. Index must be DatetimeIndex.
        要添加列的 DataFrame。索引必须是 DatetimeIndex。
    station_code : str, optional
        BSRN station abbreviation. [e.g. 'BIL'] Used if lat/lon/elev not provided.
        BSRN 站点缩写。[例如 'BIL']。未提供 lat/lon/elev 时使用。
    lat : float, optional
        Latitude. [degrees] Required for non-BSRN stations if station_code omitted.
        纬度。[度]。非 BSRN 站点且未提供 station_code 时必填。
    lon : float, optional
        Longitude. [degrees] Required for non-BSRN stations if station_code omitted.
        经度。[度]。非 BSRN 站点且未提供 station_code 时必填。
    elev : float, optional
        Elevation. [m] Required for non-BSRN stations if station_code omitted.
        海拔。[米]。非 BSRN 站点且未提供 station_code 时必填。
    variant : str, default "conus"
        NSRDB variant name: 'conus', 'full-disc', or 'aggregated'.
        NSRDB 变体名称。

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with added NSRDB columns.
        增加了 NSRDB 列的输入 DataFrame。

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
        raise ValueError("fetch_nsrdb_hf currently requires 'station_code' to fetch parquets from Hugging Face.")

    nsrdb_data = fetch_nsrdb_hf(df.index, station_code, variant=variant)
    for col in nsrdb_data.columns:
        df[col] = nsrdb_data[col]
    return df
