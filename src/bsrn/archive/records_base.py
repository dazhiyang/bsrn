"""
Shared Pydantic base classes for BSRN logical records.
BSRN 逻辑记录共用的 Pydantic 基类。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from .formatting import ArchiveFormatMixin


def _validation_callable(val_module, spec_validate_name: str):
    """
    Resolve ``LR_SPECS["validate_func"]`` to a function in ``validation``.

    Spec strings follow R / Fortran labels (e.g. ``F12.4_validateFunction``); Python
    identifiers use underscores (``F12_4_validateFunction``). Try the spec name first,
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


class ArchiveRecordBase(ArchiveFormatMixin, BaseModel):
    """
    Base LR model: archive validation (R validateFunc) + Fortran formatting.
    逻辑记录基类：存档校验（R validateFunc）与 Fortran 格式化。
    """

    model_config = ConfigDict(extra="ignore", frozen=False)

    @property
    def _private(self):
        """
        Field values as a dict for ``get_bsrn_format`` implementations.
        ``get_bsrn_format`` 实现使用的字段值字典。
        """
        return {name: getattr(self, name) for name in type(self).model_fields}

    @model_validator(mode="after")
    def _archive_validate_and_coerce(self):
        """
        Run ``validation.*_validateFunction`` for each non-None field; coerce I/F scalars.
        对非 ``None`` 字段运行 ``validation.*_validateFunction``；对标量 I/F 做强制转换。

        Coerced values are written back with ``object.__setattr__`` so this validator always
        returns ``self`` (Pydantic v2 warns when an ``after`` validator returns
        ``model_copy(update=...)`` from ``__init__``).
        """
        import bsrn.archive.validation as val_module

        cls_name = self.__class__.__name__
        ym = getattr(self, "yearMonth", None) if cls_name in ("LR0100", "LR4000") else None
        for fname in type(self).model_fields:
            meta = self._field_meta(fname)
            vfn = meta["validate_func"]
            raw = getattr(self, fname)
            if raw is None:
                continue
            fn = _validation_callable(val_module, vfn)
            try:
                if vfn in ("LR0100_validateFunction", "LR4000_validateFunction"):
                    clean = fn(raw, yearMonth=ym)
                else:
                    clean = fn(raw)
            except Exception as e:
                raise ValueError(f"{fname}\n {str(e)}") from e
            if isinstance(raw, (np.ndarray, pd.Series, list, tuple)):
                if clean is not raw:
                    object.__setattr__(self, fname, clean)
            else:
                clean = self._coerce_stored_scalar(fname, clean, meta)
                if clean != raw:
                    object.__setattr__(self, fname, clean)
        return self
