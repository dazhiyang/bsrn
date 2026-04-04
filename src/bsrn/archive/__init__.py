"""
BSRN station-to-archive: logical-record specs, Pydantic models, validators, and ASCII formatters.

- ``specs`` — ``LR_SPECS`` and supporting tables (stations, quantities, surfaces, …).
- ``records_base`` / ``records_models`` — Pydantic ``LR*`` models; BSRN checks use ``AfterValidator`` and ``field_validator`` (see ``records_models``).
- ``validation`` — Python callables referenced by each field’s ``validate_func`` string.
- ``archive_lr_formats`` — ``get_bsrn_format`` on each LR plus ``get_azimuth_elevation`` for LR0004 horizon lines.

BSRN 台站存档：逻辑记录规范、Pydantic 模型、字段校验与定宽 ASCII 格式化。
"""

from . import specs, validation
from .archive_lr_formats import get_azimuth_elevation
from .records_base import ArchiveRecordBase
from .records_models import (
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
    "specs",
    "validation",
]
