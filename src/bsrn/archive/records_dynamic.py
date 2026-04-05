"""
Dynamically built Pydantic logical-record models from ``specs.LR_SPECS``.

Each class is ``create_model(...)`` plus ``get_bsrn_format`` from ``archive_lr_formats``.
由 ``specs.LR_SPECS`` 动态构建的 Pydantic 逻辑记录模型；``get_bsrn_format`` 来自 ``archive_lr_formats``。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, create_model

from .archive_lr_formats import _FORMATTERS
from .records_base import ArchiveRecordBase
from .specs import LR_SPECS


def _field_spec(meta):
    """
    Map one ``LR_SPECS`` field entry to a Pydantic ``(type, Field)`` pair.
    将 ``LR_SPECS`` 单字段映射为 Pydantic ``(类型, Field)`` 对。
    """
    extra = {"archive": dict(meta)}
    if meta["format"] == "L":
        return bool, Field(default=False, json_schema_extra=extra)
    if meta["validate_func"] == "L_validateFunction" and meta.get("default") is False:
        return bool, Field(default=False, json_schema_extra=extra)
    return Any, Field(default=meta["default"], json_schema_extra=extra)


def _build_model(lr_code):
    """
    Create one LR model class and attach ``get_bsrn_format``.
    创建单个 LR 模型类并挂载 ``get_bsrn_format``。
    """
    fields = {}
    for fname, meta in LR_SPECS[lr_code].items():
        fields[fname] = _field_spec(meta)
    cls = create_model(
        lr_code,
        __base__=ArchiveRecordBase,
        __module__=__name__,
        **fields,
    )
    cls.get_bsrn_format = _FORMATTERS[lr_code]
    return cls


for _code in LR_SPECS:
    globals()[_code] = _build_model(_code)

__all__ = [
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
]
