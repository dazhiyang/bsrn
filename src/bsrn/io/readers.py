"""
bsrn file readers.
Handles .001, .002, ... and other archive formats.
BSRN 文件读取模块。
处理 .001, .002, ... 等存档格式。
"""

import pandas as pd
import gzip
import io
import os


def read_bsrn_station_to_archive(file_path):
    """
    Reader for BSRN station-to-archive format, specifically for LR0100 records.
    BSRN 站点存档格式读取器，专门针对 LR0100 记录。

    Parameters
    ----------
    file_path : str
        Path to the BSRN station-to-archive file (must be .dat.gz).
        BSRN 站点存档文件的路径（必须为 .dat.gz）。

    Returns
    -------
    df : pd.DataFrame or None
        Parsed data with columns: day_number, minute_number, ghi, bni, dhi, lwd, temp, rh, pressure.
        解析后的数据，包含列：day_number, minute_number, ghi, bni, dhi, lwd, temp, rh, pressure。
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

    # Process interleaved lines (2 lines per minute) / 处理交叉行（每分钟 2 行）
    # Line 1: Day, Min, Global, ..., Direct, ...
    # Line 2: Diffuse, ..., Longwave, ..., Temp, Humid, Press
    data_lines = lines[start_idx:end_idx]

    # Remove any empty or malformed trailing lines / 移除末尾空行或格式错误行
    if len(data_lines) % 2 != 0:
        data_lines = data_lines[:-1]

    # Extract year and month from filename (e.g., qiq0124.dat.gz -> Jan 2024)
    # 从文件名提取年份和月份（例如 qiq0124.dat.gz -> 2024年1月）
    filename = os.path.basename(file_path)
    try:
        # BSRN format: XXXMMYY.dat.gz
        month_str = filename[3:5]
        year_str = filename[5:7]
        year = 2000 + int(year_str)
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
    
    # Drop raw day/minute columns as they are now in the index / 删除原始天/列，因为它们现在在索引中
    df = df.drop(columns=['day', 'minute'])
    
    return df


def read_bsrn_multiple_files(directory, extension="*.dat.gz"):
    """
    Read multiple files in a directory and concatenate them.
    读取目录中的多个文件并进行合并。

    Parameters
    ----------
    directory : str
        Path to the directory containing BSRN files.
        包含 BSRN 文件的目录路径。
    extension : str, default "*.dat.gz"
        File glob pattern.
        文件匹配模式。

    Returns
    -------
    df : pd.DataFrame or None
        Concatenated data.
        合并后的数据。
    """
    from glob import glob
    files = sorted(glob(os.path.join(directory, extension)))
    
    if not files:
        print(f"No files found matching {extension} in {directory}")
        return None
        
    dfs = []
    for f in files:
        data = read_bsrn_station_to_archive(f)
        if data is not None:
            dfs.append(data)
            
    if not dfs:
        return None
        
    return pd.concat(dfs).sort_index()
