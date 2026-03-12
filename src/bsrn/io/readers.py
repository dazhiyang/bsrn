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

    line1_list = []
    line2_list = []
    
    for i in range(0, len(data_lines), 2):
        l1 = data_lines[i].split()
        l2 = data_lines[i+1].split()

        # We need specific columns as per R script logic / 根据 R 脚本逻辑提取特定列
        # Python indices (0-based):
        # Line 1: 0(day), 1(min), 2(ghi), 6(bni)
        # Line 2: 0(dhi), 4(lwd), 8(temp), 9(rh), 10(pressure)
        
        if len(l1) >= 7 and len(l2) >= 11:
            row = [
                float(l1[0]),   # day_number
                float(l1[1]),   # minute_number
                float(l1[2]),   # ghi
                float(l1[6]),   # bni
                float(l2[0]),   # dhi
                float(l2[4]),   # lwd
                float(l2[8]),   # temp
                float(l2[9]),   # rh
                float(l2[10])   # pressure
            ]
            line1_list.append(row)

    columns = ["day_number", "minute_number", "ghi", "bni", "dhi", "lwd", "temp", "rh", "pressure"]
    df = pd.DataFrame(line1_list, columns=columns)

    # Replace BSRN missing value indicators (-1) with NaN
    # 将 BSRN 缺失值标记 (-1) 替换为 NaN
    # Note: For some variables, -1 might be valid, but typically in LR0100 it means missing.
    # 注意：对于某些变量，-1 可能是有效的，但在 LR0100 中通常表示缺失。
    # df = df.replace(-1.0, float('nan'))
    
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
        
    return pd.concat(dfs, ignore_index=True)
