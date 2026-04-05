"""
I/O subpackage: BSRN file readers, FTP retrieval, and auxiliary radiation sources.
I/O 子包：BSRN 文件读取、FTP 获取及辅助辐射数据源。

Modules expose readers (:mod:`reader`), FTP helpers (:mod:`retrieval`), CAMS CRS /
McClear / MERRA-2 / NSRDB integrations, and thin Hugging Face fetch paths.
各模块提供读取器（:mod:`reader`）、FTP 工具（:mod:`retrieval`）、CAMS CRS / McClear /
MERRA-2 / NSRDB 集成及 Hugging Face 拉取封装。
"""

from . import crs, reader, retrieval, mcclear, merra2, nsrdb
