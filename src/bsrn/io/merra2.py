"""
MERRA-2 parquet loader for REST2 clear-sky inputs.
用于 REST2 晴空输入的 MERRA-2 parquet 加载器。

Fetches MERRA-2 parquet from Hugging Face (dazhiyang/bsrn-merra2) into RAM (no disk cache).
从 Hugging Face (dazhiyang/bsrn-merra2) 获取 MERRA-2 parquet 到内存（无磁盘缓存）。
"""

import io
import pandas as pd
import requests

from bsrn.constants import MERRA2_HF_REPO_ID, HF_MAINTAINER_EMAIL


def _hf_fetch_to_memory(repo_id, filename):
    """
    Fetch one dataset file from Hugging Face into RAM (no disk cache; used by ``_fetch_merra2_from_hf``).
    从 Hugging Face 拉取单个数据集文件到内存（无磁盘缓存；由 ``_fetch_merra2_from_hf`` 使用）。

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset repo id.
        Hugging Face 数据集仓库 id。
    filename : str
        Path within the repo (e.g. ``stn/stnMMYY_merra2.parquet``).
        仓库内路径（如 ``stn/stnMMYY_merra2.parquet``）。

    Returns
    -------
    content : bytes
        Raw file bytes.
        原始文件字节。

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
        未安装 ``huggingface_hub`` 时。
    FileNotFoundError
        On 404 or other fetch failure; message includes maintainer contact.
        404 或其他获取失败时；信息含维护者联系方式。
    """
    try:
        from huggingface_hub import hf_hub_url
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for MERRA-2. Install with: pip install huggingface_hub"
        )
    print(f"Fetching MERRA-2 from Hugging Face: {filename}")
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


def _fetch_merra2_from_hf(station_code, index):
    """
    Fetch one parquet per unique (year, month) in *index* from Hugging Face (used by ``fetch_rest2``).
    按 *index* 中唯一 (年, 月) 从 Hugging Face 各取一个 parquet（由 ``fetch_rest2`` 使用）。

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
        One element per month, in sorted (year, month) order.
        每月一个元素，按 (年, 月) 排序。

    Raises
    ------
    ValueError
        If *index* is empty.
        *index* 为空时。
    """
    if index.empty:
        raise ValueError("index must not be empty. / index 不能为空。")
    stn = station_code.lower()
    # Use (year, month) pairs to avoid PeriodIndex (which drops tz and triggers a warning)
    unique_pairs = set(zip(index.year, index.month))
    contents = []
    for year, month in sorted(unique_pairs):
        filename = f"{stn}{month:02d}{year % 100:02d}_merra2.parquet"
        hf_filename = f"{stn}/{filename}"
        content = _hf_fetch_to_memory(MERRA2_HF_REPO_ID, hf_filename)
        contents.append(content)
    return contents


def _load_merra2_parquet(path_or_bytes):
    """
    Load one MERRA-2 parquet into a UTC-indexed DataFrame (used by ``fetch_rest2`` only).
    将单个 MERRA-2 parquet 加载为 UTC 索引 DataFrame（仅由 ``fetch_rest2`` 使用）。

    Parameters
    ----------
    path_or_bytes : str, path-like, bytes, or file-like
        Path to the parquet file, or bytes content, or file-like object.
        parquet 文件路径、字节内容或类文件对象。

    Returns
    -------
    data : pd.DataFrame
        MERRA-2 data with columns AOD55, ALPHA, ALBEDO, TQV, TO3, PS
        and UTC DatetimeIndex.
        含 AOD55、ALPHA、ALBEDO、TQV、TO3、PS 列及 UTC DatetimeIndex 的 MERRA-2 数据。
    """
    if isinstance(path_or_bytes, bytes):
        path_or_bytes = io.BytesIO(path_or_bytes)
    data = pd.read_parquet(path_or_bytes)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "MERRA-2 parquet must have DatetimeIndex. / MERRA-2 parquet 必须有 DatetimeIndex。"
        )
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    return data


def _parse_merra2_for_rest2(raw, target_index):
    """
    Reindex, interpolate, derive BETA, convert units for REST2 (used by ``fetch_rest2`` only).
    为重索引、插值、推导 BETA、单位换算以适配 REST2（仅由 ``fetch_rest2`` 使用）。

    Parameters
    ----------
    raw : pd.DataFrame
        MERRA-2 data with columns AOD55, ALPHA, ALBEDO, TQV, TO3, PS.
        含 AOD55、ALPHA、ALBEDO、TQV、TO3、PS 列的 MERRA-2 数据。
    target_index : pd.DatetimeIndex
        Target 1-min index, typically the BSRN time index.
        目标 1 分钟时间索引，通常为 BSRN 时间索引。

    Returns
    -------
    rest2 : pd.DataFrame
        REST2-ready DataFrame with columns PS, ALBEDO, ALPHA, BETA, TO3, TQV.
        REST2 就绪的 DataFrame，含 PS、ALBEDO、ALPHA、BETA、TO3、TQV 列。
    """
    if not isinstance(target_index, pd.DatetimeIndex):
        raise ValueError(
            "target_index must be a DatetimeIndex. / target_index 必须为 DatetimeIndex。"
        )

    df = raw.copy()

    # Reindex to 1-min target index and interpolate / 重索引到 1 分钟目标并插值
    df = df.reindex(target_index)
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")

    # Unit conversions / 单位换算
    # PS: Pa → hPa (÷ 100)
    if "PS" in df.columns:
        df["PS"] = df["PS"] / 100.0
    # TQV: kg/m² → atm.cm (÷ 10)
    if "TQV" in df.columns:
        df["TQV"] = df["TQV"] / 10.0
    # TO3: Dobson → atm.cm (÷ 1000)
    if "TO3" in df.columns:
        df["TO3"] = df["TO3"] / 1000.0

    # Derive BETA = AOD55 * 0.55^ALPHA; replace AOD55=0 with 0.001 before
    # 推导 BETA = AOD55 * 0.55^ALPHA；计算前将 AOD55=0 替换为 0.001
    if {"AOD55", "ALPHA"}.issubset(df.columns):
        aod55 = df["AOD55"].replace(0.0, 0.001)
        df["BETA"] = aod55 * (0.55 ** df["ALPHA"])
        df = df.drop(columns=["AOD55"])

    return df


def fetch_rest2(index, station_code):
    """
    Fetch MERRA-2 from Hugging Face and return REST2-ready inputs aligned to target index.
    Parallel to fetch_mcclear; both return aligned clear-sky inputs.
    从 Hugging Face 获取 MERRA-2 并返回对齐到目标索引的 REST2 就绪输入。与 fetch_mcclear 对应。

    Parameters
    ----------
    index : pd.DatetimeIndex
        Target time index to align REST2 outputs to.
        需要对齐的目标时间索引。
    station_code : str
        BSRN station code (e.g. "QIQ").
        BSRN 站点代码（如 "QIQ"）。

    Returns
    -------
    aligned : pd.DataFrame
        REST2 inputs reindexed to `index` with columns PS, ALBEDO, ALPHA, BETA, TO3, TQV.
        重新索引到 `index` 的 REST2 输入，含 PS、ALBEDO、ALPHA、BETA、TO3、TQV 列。
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            "index must be a pandas DatetimeIndex. / index 必须是 pandas DatetimeIndex。"
        )
    if index.empty:
        raise ValueError("index must not be empty. / index 不能为空。")

    contents = _fetch_merra2_from_hf(station_code, index)
    dfs = [_load_merra2_parquet(c) for c in contents]
    raw = pd.concat(dfs).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    rest2 = _parse_merra2_for_rest2(raw, index)
    return rest2
