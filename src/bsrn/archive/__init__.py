"""
BSRN archive: logical-record specs, validation, and R-style helpers.

R reference layout under ``data/R``:

- ``0_data.R`` — table placeholders (A1–A7); Python: ``specs`` (``LR_SPECS`` + station / code tables)
- ``1_utils.R`` — calendar helpers and bindings; Python: ``api`` (``get_azimuth_elevation``, …; month length via ``calendar.monthrange`` where needed)
- ``0_bsrnFormats_*.R`` — format strings; Python: ``records_dynamic`` (``LR*``) + ``get_bsrn_format`` on each model
- ``1_validateFunc_*.R`` — Python: ``validation``
- ``2_R6Class_*.R`` — Python: ``records_dynamic`` (``LR0001`` … ``LR4000``); ``api`` (``get_azimuth_elevation``)

BSRN 存档：逻辑记录规范、校验与 R 风格辅助函数。上述 R 文件与 Python 子模块对应关系见源码注释。
"""

from . import api, specs, validation
from .api import get_azimuth_elevation
from .records_base import ArchiveRecordBase
from .records_dynamic import (
    LR0001,
    LR0002,
    LR0003,
    LR0004,
    LR0005,
    LR0006,
    LR0007,
    LR0008,
    LR0100,
    LR4000,
    LR4000CONST,
)
from .specs import (
    LR_SPECS,
    PYRGEOMETER_BODY,
    PYRGEOMETER_DOME,
    QUANTITIES,
    STATION_METADATA,
    SURFACES,
    TOPOGRAPHIES,
)

__all__ = [
    "LR_SPECS",
    "STATION_METADATA",
    "TOPOGRAPHIES",
    "SURFACES",
    "QUANTITIES",
    "PYRGEOMETER_BODY",
    "PYRGEOMETER_DOME",
    "ArchiveRecordBase",
    "LR0001",
    "LR0002",
    "LR0003",
    "LR0004",
    "LR0005",
    "LR0006",
    "LR0007",
    "LR0008",
    "LR0100",
    "LR4000",
    "LR4000CONST",
    "get_azimuth_elevation",
    "api",
    "specs",
    "validation",
]
