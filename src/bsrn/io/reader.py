"""
bsrn file readers.
BSRN 文件读取模块。

Handles .001, .002, ... and other archive formats.
处理 .001, .002, ... 等存档格式。
"""

import gzip
import os
import pandas as pd


def read_lr0100(file_path):
    """
    BSRN station-to-archive format reader [1]_, specifically for LR0100 records.
    BSRN 站点存档格式读取器，专门针对 LR0100 记录。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (must be .dat.gz).
        BSRN 站点存档文件的路径（必须为 .dat.gz）。

    Returns
    -------
    df : pd.DataFrame or None
        Parsed data with columns: ghi, bni, dhi, lwd [W/m^2], temp [C], rh [%], pressure [hPa].
        解析后的数据，列包含：ghi, bni, dhi, lwd [瓦/平方米], temp [摄氏度], rh [%], pressure [百帕]。

    References
    ----------
    .. [1] Driemel, A., et al. (2018). Baseline Surface Radiation Network (BSRN): 
       structure and data description (1992–2017). Earth System Science Data, 
       10(3), 1491-1501.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    if not file_path.endswith('.dat.gz'):
        print(f"Error: Only .dat.gz files are supported: {file_path}")
        return None

    # Open the gzipped file / 打开 gzip 压缩文件
    with gzip.open(file_path, 'rt', encoding='ascii') as f:
        lines = f.readlines()

    # Find the start of LR0100 record / 查找 LR0100 记录的开始
    # Record markers start with '*' / 记录标记以 '*' 开头
    lr_starts = [i for i, line in enumerate(lines) if line.startswith('*')]

    # Locate *U0100 or *C0100 / 定位 *U0100 或 *C0100
    pos = -1
    for i, idx in enumerate(lr_starts):
        marker = lines[idx].strip()
        if marker.endswith('0100'):
            pos = i
            break
    
    if pos == -1:
        print("Error: LR0100 record not found in file.")
        return None

    start_idx = lr_starts[pos] + 1
    # End is the next marker or end of file / 结束于下一个标记或文件末尾
    if pos + 1 < len(lr_starts):
        end_idx = lr_starts[pos + 1]
    else:
        end_idx = len(lines)

    data_lines = lines[start_idx:end_idx]

    # Remove any empty or malformed trailing lines / 移除末尾空行或格式错误行
    if len(data_lines) % 2 != 0:
        data_lines = data_lines[:-1]

    # Extract year and month from filename (e.g., qiq0124.dat.gz -> Jan 2024)
    # 从文件名提取年份和月份（例如 qiq0124.dat.gz -> 2024年1月）
    filename = os.path.basename(file_path)
    try:
        # BSRN format: XXXMMYY.dat.gz. BSRN started 1991, so yy>=91 -> 1900s.
        month_str = filename[3:5]
        year_str = filename[5:7]
        yy = int(year_str)
        year = (1900 + yy) if yy >= 91 else (2000 + yy)
        month = int(month_str)
    except Exception as e:
        print(f"Error: Could not parse year/month from filename {filename}: {e}")
        return None

    data_list = []
    for i in range(0, len(data_lines), 2):
        l1 = data_lines[i].split()
        l2 = data_lines[i+1].split()

        # Extract specific columns for radiation and meteorology / 提取特定的辐射和气象参数列
        # Python indices (0-based):
        # Line 1: 0(day), 1(min), 2(ghi mean), 6(bni mean)
        # Line 2: 0(dhi mean), 4(lwd mean), 8(temp), 9(rh), 10(pressure)
        if len(l1) >= 7 and len(l2) >= 11:
            row = [
                int(l1[0]),     # day
                int(l1[1]),     # minute-of-day (0-1439)
                float(l1[2]),   # ghi
                float(l1[6]),   # bni
                float(l2[0]),   # dhi
                float(l2[4]),   # lwd
                float(l2[8]),   # temp
                float(l2[9]),   # rh
                float(l2[10])   # pressure
            ]
            data_list.append(row)

    columns = ["day", "minute", "ghi", "bni", "dhi", "lwd", "temp", "rh", "pressure"]
    df = pd.DataFrame(data_list, columns=columns)


    # Convert day and minute to DatetimeIndex / 将天和分钟转换为 DatetimeIndex
    # Note: minute is 'minute after midnight' (0-1439)
    try:
        # Create base date for the month in UTC / 创建当月基础日期（UTC 时区）
        base_date = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        # Calculate time offset / 计算时间偏移
        # Timestamps = base_date + (day-1) days + minute minutes
        # 时间戳 = 基础日期 + (天数-1) + 分钟数
        times = base_date + pd.to_timedelta(df['day'] - 1, unit='D') + \
                pd.to_timedelta(df['minute'], unit='m')
        df.index = times
        df.index.name = 'time'
    except Exception as e:
        print(f"Error: Datetime conversion failed for {filename}: {e}")
        # Continue with numeric index if conversion fails / 如果转换失败，继续使用数值索引

    # Replace BSRN missing value indicators with NaN / 将 BSRN 缺失值标记替换为 NaN
    # Missing codes per specification: -999, -99.9 / 规范定义的缺失值：-999, -99.9
    for val in [-999.0, -99.9]:
        df = df.replace(val, float('nan'))
    
    # Drop raw day/minute columns as they are now in the index / 删除原始天/分钟列，因为它们现在在索引中
    df = df.drop(columns=['day', 'minute'])
    
    return df


def read_lr0300(file_path):
    """
    Reader for BSRN station-to-archive format, specifically for LR0300 records.
    BSRN 站点存档格式读取器，专门针对 LR0300 记录。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (must be .dat.gz).
        BSRN 站点存档文件的路径（必须为 .dat.gz）。

    Returns
    -------
    df : pd.DataFrame or None
        Parsed data with columns: swu, lwu, net [W/m^2].
        解析后的数据，列包含：swu, lwu, net [瓦/平方米]。
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    if not file_path.endswith('.dat.gz'):
        print(f"Error: Only .dat.gz files are supported: {file_path}")
        return None

    # Open the gzipped file / 打开 gzip 压缩文件
    with gzip.open(file_path, 'rt', encoding='ascii') as f:
        lines = f.readlines()

    # Find the start of LR0300 record / 查找 LR0300 记录的开始
    # Record markers start with '*' / 记录标记以 '*' 开头
    lr_starts = [i for i, line in enumerate(lines) if line.startswith('*')]

    # Locate *U0300 or *C0300 / 定位 *U0300 或 *C0300
    pos = -1
    for i, idx in enumerate(lr_starts):
        marker = lines[idx].strip()
        if marker.endswith('0300'):
            pos = i
            break
    
    if pos == -1:
        print("Error: LR0300 record not found in file.")
        return None

    start_idx = lr_starts[pos] + 1
    # End is the next marker or end of file / 结束于下一个标记或文件末尾
    if pos + 1 < len(lr_starts):
        end_idx = lr_starts[pos + 1]
    else:
        end_idx = len(lines)

    data_lines = lines[start_idx:end_idx]

    # Extract year and month from filename (e.g., qiq0124.dat.gz -> Jan 2024)
    # 从文件名提取年份和月份（例如 qiq0124.dat.gz -> 2024年1月）
    filename = os.path.basename(file_path)
    try:
        # BSRN format: XXXMMYY.dat.gz. BSRN started 1991, so yy>=91 -> 1900s.
        month_str = filename[3:5]
        year_str = filename[5:7]
        yy = int(year_str)
        year = (1900 + yy) if yy >= 91 else (2000 + yy)
        month = int(month_str)
    except Exception as e:
        print(f"Error: Could not parse year/month from filename {filename}: {e}")
        return None

    data_list = []
    # Process lines (1 line per minute) / 处理数据行（每分钟 1 行）
    for line in data_lines:
        l = line.split()
        
        # Ensure enough columns exist (Day + Min + 3 blocks of 4 variables = 14 columns)
        # 确保存在足够的数据列（日+分钟+3个区块*4个变量 = 14列）
        if len(l) >= 14:
            try:
                # Extract mean values for SWU, LWU, and Net radiation
                # 提取上行短波、上行长波和净辐射的平均值
                # Python indices (0-based):
                # 0(day), 1(min), 2(swu mean), 6(lwu mean), 10(net mean)
                row = [
                    int(l[0]),     # day
                    int(l[1]),     # minute
                    float(l[2]),   # swu
                    float(l[6]),   # lwu
                    float(l[10])   # net
                ]
                data_list.append(row)
            except ValueError:
                # Skip lines that cannot be parsed as numeric / 跳过无法解析为数字的行
                continue

    columns = ["day", "minute", "swu", "lwu", "net"]
    df = pd.DataFrame(data_list, columns=columns)

    if df.empty:
        return df

    # Convert day and minute to DatetimeIndex / 将天和分钟转换为 DatetimeIndex
    # Note: minute is 'minute after midnight' (0-1439)
    try:
        # Create base date for the month in UTC / 创建当月基础日期（UTC 时区）
        base_date = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        # Calculate time offset / 计算时间偏移
        # Timestamps = base_date + (day-1) days + minute minutes
        # 时间戳 = 基础日期 + (天数-1) + 分钟数
        times = base_date + pd.to_timedelta(df['day'] - 1, unit='D') + \
                pd.to_timedelta(df['minute'], unit='m')
        df.index = times
        df.index.name = 'time'
    except Exception as e:
        print(f"Error: Datetime conversion failed for {filename}: {e}")
        # Continue with numeric index if conversion fails / 如果转换失败，继续使用数值索引

    # Replace BSRN missing value indicators with NaN / 将 BSRN 缺失值标记替换为 NaN
    # Missing codes per specification: -999, -99.9 / 规范定义的缺失值：-999, -99.9
    for val in [-999.0, -99.9]:
        df = df.replace(val, float('nan'))
    
    # Drop raw day/minute columns as they are now in the index / 删除原始天/分钟列，因为它们现在在索引中
    df = df.drop(columns=['day', 'minute'])
    
    return df



def read_lr4000(file_path):
    """
    Reader for BSRN station-to-archive format, specifically for LR4000 records 
    (pyrgeometer temperatures).
    BSRN 站点存档格式读取器，专门针对 LR4000 记录（长波辐射表温度）。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (must be .dat.gz).
        BSRN 站点存档文件的路径（必须为 .dat.gz）。

    Returns
    -------
    df : pd.DataFrame or None
        Parsed data with columns for downward (``td_``) and upward (``tu_``) 
        pyrgeometer dome, body, and thermopile (tp) measurements.
        解析后的数据，包含下行 (``td_``) 和上行 (``tu_``) 长波辐射表罩温、体温和热电堆 (tp) 测量值。
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    if not file_path.endswith('.dat.gz'):
        print(f"Error: Only .dat.gz files are supported: {file_path}")
        return None

    # Open the gzipped file / 打开 gzip 压缩文件
    with gzip.open(file_path, 'rt', encoding='ascii') as f:
        lines = f.readlines()

    # Find the start of LR4000 record / 查找 LR4000 记录的开始
    lr_starts = [i for i, line in enumerate(lines) if line.startswith('*')]

    # Locate *U4000 or *C4000 / 定位 *U4000 或 *C4000
    pos = -1
    for i, idx in enumerate(lr_starts):
        marker = lines[idx].strip()
        # Strictly match markers ending in '4000' / 严格匹配以 '4000' 结尾的标记
        if marker.endswith('4000'):
            pos = i
            break
    
    if pos == -1:
        return None

    start_idx = lr_starts[pos] + 1
    # End is the next marker or end of file / 结束于下一个标记或文件末尾
    if pos + 1 < len(lr_starts):
        end_idx = lr_starts[pos + 1]
    else:
        end_idx = len(lines)

    data_lines = lines[start_idx:end_idx]

    # Extract year and month from filename. BSRN started 1991, so yy>=91 -> 1900s.
    filename = os.path.basename(file_path)
    try:
        month_str = filename[3:5]
        year_str = filename[5:7]
        yy = int(year_str)
        year = (1900 + yy) if yy >= 91 else (2000 + yy)
        month = int(month_str)
    except Exception as e:
        print(f"Error: Could not parse year/month from filename {filename}: {e}")
        return None

    data_list = []
    # Process lines (1 line per minute) / 处理数据行（每分钟 1 行）
    for line in data_lines:
        l = line.split()
        
        # Expected format tokens: Day, Min, 5 down (dome1-3, body, tp), 5 up (dome1-3, body, tp)
        # 预期的标记：日期、分钟、下行 5 个（罩温 1-3、体温、热电堆）、上行 5 个
        if len(l) >= 12:
            try:
                row = [
                    int(l[0]),     # day
                    int(l[1]),     # minute
                    float(l[2]),   # td_dome1
                    float(l[3]),   # td_dome2
                    float(l[4]),   # td_dome3
                    float(l[5]),   # td_body
                    float(l[6]),   # td_tp
                    float(l[7]),   # tu_dome1
                    float(l[8]),   # tu_dome2
                    float(l[9]),   # tu_dome3
                    float(l[10]),  # tu_body
                    float(l[11])   # tu_tp
                ]
                data_list.append(row)
            except ValueError:
                continue

    cols = ["day", "minute", 
            "td_dome1", "td_dome2", "td_dome3", "td_body", "td_tp",
            "tu_dome1", "tu_dome2", "tu_dome3", "tu_body", "tu_tp"]
    df = pd.DataFrame(data_list, columns=cols)

    if df.empty:
        return df

    # Convert day and minute to DatetimeIndex / 将天和分钟转换为 DatetimeIndex
    try:
        base_date = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        times = base_date + pd.to_timedelta(df['day'] - 1, unit='D') + \
                pd.to_timedelta(df['minute'], unit='m')
        df.index = times
        df.index.name = 'time'
    except Exception as e:
        print(f"Error: Datetime conversion failed for {filename}: {e}")

    # Replace BSRN missing value indicators with NaN / 将 BSRN 缺失值标记替换为 NaN
    # Missing codes per updated specification: -999.9, -99.99
    # 按照更新后的规范：缺失值代码为 -999.9, -99.99
    for val in [-999.9, -99.99]:
        df = df.replace(val, float('nan'))
    
    # Drop raw day/minute columns / 删除原始列
    df = df.drop(columns=['day', 'minute'])
    
    return df


def read_station_to_archive(file_path, logical_records="lr0100"):
    """
    Wrapper for reading multiple logical records from a BSRN station-to-archive file.
    BSRN 站点存档格式读取包装器，可读取多个逻辑记录。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (must be .dat.gz).
        BSRN 站点存档文件的路径（必须为 .dat.gz）。
    logical_records : str or list of str, default "lr0100"
        Logical record(s) to read. Options: "lr0100", "lr0300", "lr4000".
        要读取的一个或多个逻辑记录。选项："lr0100", "lr0300", "lr4000"。

    Returns
    -------
    df : pd.DataFrame or None
        Combined DataFrame containing data from all requested logical records.
        包含所有请求的逻辑记录数据的合并 DataFrame。
    """
    if isinstance(logical_records, str):
        logical_records = [logical_records]
    
    # Map supported records to their reader functions
    reader_map = {
        "lr0100": read_lr0100,
        "lr0300": read_lr0300,
        "lr4000": read_lr4000
    }
    
    dfs = []
    for lr in logical_records:
        lr_key = lr.lower()
        if lr_key in reader_map:
            df_lr = reader_map[lr_key](file_path)
            if df_lr is not None:
                dfs.append(df_lr)
        else:
            print(f"Error: Unsupported logical record '{lr}'.")
            
    if not dfs:
        return None
        
    # Merge all DataFrames
    result = dfs[0]
    for next_df in dfs[1:]:
        result = result.combine_first(next_df)
    
    return result
