"""
Shared Pydantic base classes for BSRN logical records.
BSRN 逻辑记录共用的 Pydantic 基类。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from .formatting import ArchiveFormatMixin
from .specs import LR_SPECS


def _validation_callable(val_module, spec_validate_name: str):
    """
    Resolve ``LR_SPECS["validate_func"]`` to a function in ``validation``.

    Names follow the BSRN archive convention (e.g. ``F12.4_validateFunction``); Python
    identifiers use underscores (``F12_4_validateFunction``). Try the spec string first,
    then the underscored form.
    """
    fn = getattr(val_module, spec_validate_name, None)
    if fn is not None:
        return fn
    py_name = spec_validate_name.replace(".", "_")
    if py_name != spec_validate_name:
        return getattr(val_module, py_name)
    raise AttributeError(
        f"bsrn.archive.validation has no {spec_validate_name!r} (or {py_name!r})"
    )


def make_archive_after_validator(lr_code: str, field_name: str):
    """
    Build a unary callable for :class:`pydantic.functional_validators.AfterValidator`.

    Reads ``LR_SPECS[lr_code][field_name]`` (``validate_func`` + ``format``). Minute
    LR0100 / LR4000 columns must use :func:`field_validator` with ``yearMonth`` instead.
    """
    meta = LR_SPECS[lr_code][field_name]
    vfn = meta["validate_func"]
    if vfn in ("LR0100_validateFunction", "LR4000_validateFunction"):
        raise ValueError(
            f"{lr_code}.{field_name}: use field_validator with yearMonth for minute columns"
        )

    def validate(value):
        if value is None:
            return value
        import bsrn.archive.validation as val_module

        fn = _validation_callable(val_module, vfn)
        try:
            clean = fn(value)
        except Exception as e:
            raise ValueError(f"{field_name}\n {str(e)}") from e
        if isinstance(value, (np.ndarray, pd.Series, list, tuple)):
            return clean if clean is not value else value
        return ArchiveFormatMixin._coerce_stored_scalar(field_name, clean, meta)

    return validate


class ArchiveRecordBase(ArchiveFormatMixin, BaseModel):
    """
    Base class for archive LRs: Pydantic ``BaseModel`` plus ``ArchiveFormatMixin`` for
    Fortran-width ASCII output. Subclasses attach BSRN checks via
    :class:`pydantic.functional_validators.AfterValidator` and, for LR0100 / LR4000
    minute vectors, :func:`pydantic.field_validator` (with ``ValidationInfo`` for
    ``yearMonth``).
    逻辑记录基类：Pydantic 模型 + 定宽格式化；子类用 ``AfterValidator`` / ``field_validator`` 挂接 BSRN 校验。
    """

    model_config = ConfigDict(extra="ignore", frozen=False)

    @property
    def _private(self):
        """
        Field values as a dict for ``get_bsrn_format`` implementations.
        ``get_bsrn_format`` 实现使用的字段值字典。
        """
        return {name: getattr(self, name) for name in type(self).model_fields}
