"""
bsrn data retrieval module.
BSRN 数据获取模块。

Handles FTP connections and automated downloads.
处理 FTP 连接并进行自动下载。
"""

import os
import time
from ftplib import FTP


def get_bsrn_file_inventory(stations, username, password, host="ftp.bsrn.awi.de"):
    """
    Connects to bsrn ftp and lists available station-to-archive files.
    连接到 BSRN FTP 并列出可用的站点存档文件。

    Parameters
    ----------
    stations : list
        List of station abbreviations (e.g., ['PAY', 'NYA']).
        站点缩写列表（例如 ['PAY', 'NYA']）。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。
    host : str, default "ftp.bsrn.awi.de"
        FTP host address.
        FTP 主机地址。

    Returns
    -------
    inventory : dict
        Mapping of station abbreviations to lists of filenames.
        站点缩写到文件名列表的映射。

    References
    ----------
    .. [1] Driemel, A., et al. (2018). Baseline Surface Radiation Network (BSRN): 
       structure and data description (1992–2017). Earth System Science Data, 
       10(3), 1491-1501.
    """
    inventory = {}
    try:
        with FTP(host) as ftp:
            ftp.set_pasv(True)
            ftp.login(user=username, passwd=password)

            for i, stn in enumerate(stations):
                print(f"[{i+1}/{len(stations)}] Fetching inventory for station {stn.upper()}...")
                stn_lower = stn.lower()
                success = False

                # Retry logic for the connection / 连接重试逻辑
                for attempt in range(2):
                    try:
                        ftp.cwd("/")
                        ftp.cwd(stn_lower)

                        inventory[stn.upper()] = ftp.nlst()
                        success = True
                        break
                    except Exception as e:
                        if attempt == 0:
                            # Re-establish connection on failure / 失败时重新建立连接
                            try:
                                ftp.connect(host)
                                ftp.login(user=username, passwd=password)
                                ftp.set_pasv(True)
                            except:
                                # Connection failed / 连接失败
                                pass
                        else:
                            print(f"BSRN FTP: Failed to retrieve {stn} after retry: {e}")

                if not success:
                    inventory[stn.upper()] = []

    except Exception as e:
        print(f"BSRN FTP: Major Connection Error: {e}")
    return inventory


def download_bsrn_single(station, year, month, local_dir, username, password, host="ftp.bsrn.awi.de"):
    """
    Download a single BSRN file by specifying station, year, and month.
    通过指定站点、年份和月份下载单个 BSRN 文件。

    Parameters
    ----------
    station : str
        Station abbreviation (e.g., 'pay').
        站点缩写（例如 'pay'）。
    year : int or str
        Four-digit year (e.g., 2024 or '2024').
        四位年份（例如 2024 或 '2024'）。
    month : int or str
        Month number or string (1-12 or '01'-'12').
        月份（1-12 或 '01'-'12'）。
    local_dir : str
        The local directory to save the file.
        保存文件的本地目录。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。
    host : str, default "ftp.bsrn.awi.de"
        FTP host address.
        FTP 主机地址。

    Returns
    -------
    local_path : str or None
        The path to the downloaded file, or None if failed.
        下载文件的路径，如果失败则返回 None。
    """
    # BSRN filenames use 2-digit months and 2-digit years, strictly lowercase
    # BSRN 文件名使用 2 位月份和 2 位年份，且严格小写
    year_str = str(year)[-2:]
    month_int = int(month)
    filename = f"{station.lower()}{month_int:02d}{year_str}.dat.gz"
    return download_bsrn_files([filename], local_dir, username, password, host=host)[0]


def download_bsrn_stn(station, local_dir, username, password, host="ftp.bsrn.awi.de"):
    """
    Download all available station-to-archive files for a specific station.
    下载特定站点的所有可用站点存档文件。

    Parameters
    ----------
    station : str
        Station abbreviation (e.g., 'pay').
        站点缩写（例如 'pay'）。
    local_dir : str
        The local directory to save the files.
        保存文件的本地目录。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。
    host : str, default "ftp.bsrn.awi.de"
        FTP host address.
        FTP 主机地址。

    Returns
    -------
    downloaded_paths : list
        List of paths to the downloaded files.
        已下载文件的路径列表。
    """
    inventory = get_bsrn_file_inventory([station], username, password, host=host)
    filenames = inventory.get(station.upper(), [])
    return download_bsrn_files(filenames, local_dir, username, password, host=host)


def download_bsrn_mon(stations, year, month, local_dir, username, password, host="ftp.bsrn.awi.de"):
    """
    Download station-to-archive files for multiple stations for a specific month and year.
    下载特定月份和年份的多个站点的站点存档文件。

    Parameters
    ----------
    stations : list of str
        List of station abbreviations (e.g., ['pay', 'nya']).
        站点缩写列表（例如 ['pay', 'nya']）。
    year : int or str
        Four-digit year (e.g., 2024 or '2024').
        四位年份（例如 2024 或 '2024'）。
    month : int or str
        Month number or string (1-12 or '01'-'12').
        月份（1-12 或 '01'-'12'）。
    local_dir : str
        The local directory to save the files.
        保存文件的本地目录。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。
    host : str, default "ftp.bsrn.awi.de"
        FTP host address.
        FTP 主机地址。

    Returns
    -------
    downloaded_paths : list
        List of paths to the downloaded files.
        已下载文件的路径列表。
    """
    year_str = str(year)[-2:]
    month_int = int(month)
    filenames = [f"{stn.lower()}{month_int:02d}{year_str}.dat.gz" for stn in stations]
    return download_bsrn_files(filenames, local_dir, username, password, host=host)


def download_bsrn_files(filenames, local_dir, username, password, host="ftp.bsrn.awi.de", retries=3):
    """
    Download many BSRN files efficiently using a single FTP connection with robust retries.
    使用单个 FTP 连接高效地下载多个 BSRN 文件，并带有稳健的重试机制。

    Parameters
    ----------
    filenames : list of str
        List of filenames to download (e.g., ['pay0123.dat.gz']).
        要下载的文件名列表（例如 ['pay0123.dat.gz']）。
    local_dir : str
        The local directory to save the files.
        保存文件的本地目录。
    username : str
        BSRN FTP username.
        BSRN FTP 用户名。
    password : str
        BSRN FTP password.
        BSRN FTP 密码。
    host : str, default "ftp.bsrn.awi.de"
        FTP host address.
        FTP 主机地址。
    retries : int, default 3
        Number of retry attempts for both connection and file transfer errors.
        连接和文件传输错误的重试次数。

    Returns
    -------
    downloaded_paths : list
        List of paths to the downloaded files.
        已下载文件的路径列表。
    """
    os.makedirs(local_dir, exist_ok=True)
    downloaded_paths = []

    def connect_ftp():
        ftp = FTP(host)
        ftp.set_pasv(True)
        ftp.login(user=username, passwd=password)
        return ftp

    ftp = None
    try:
        ftp = connect_ftp()
        for filename in filenames:
            filename_lower = filename.lower()
            local_path = os.path.join(local_dir, filename_lower)
            station_code = filename_lower[:3]

            success = False
            for attempt in range(retries):
                try:
                    if ftp is None:
                        ftp = connect_ftp()
                    
                    ftp.cwd("/")
                    ftp.cwd(station_code)

                    with open(local_path, "wb") as f:
                        ftp.retrbinary(f"RETR {filename_lower}", f.write)
                    
                    downloaded_paths.append(local_path)
                    success = True
                    break
                except Exception as e:
                    print(f"BSRN FTP: Attempt {attempt+1} failed for {filename}: {e}")
                    # Close and set ftp to None for reconnection on next attempt
                    # 关闭并将 ftp 设置为 None 以便在下次尝试时重新连接
                    try:
                        ftp.quit()
                    except:
                        # quit failed / 退出失败
                        pass
                    ftp = None
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff / 指数回退

            if not success:
                downloaded_paths.append(None)

    except Exception as e:
        print(f"BSRN FTP: Major connection error: {e}")
        while len(downloaded_paths) < len(filenames):
            downloaded_paths.append(None)
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass

    return downloaded_paths
