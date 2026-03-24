"""
BSRN station-to-archive base record class (R ``1_utils.R`` + R6 generics).

``BSRNRecord`` holds assignment validation and Fortran padding; concrete logical-record
classes (``LR0001``, …) and ``get_bsrn_format`` output live in ``formatter``.

BSRN 站点存档基类（``1_utils.R`` 与 R6 泛型逻辑）。
``BSRNRecord`` 负责赋值校验与 Fortran 填充；具体 ``LR0001`` 等及 ``get_bsrn_format`` 在 ``formatter``。

Field rules come from ``specs.LR_SPECS``. BSRN on-disk column names (e.g. ``ghi_avg``)
are format identifiers, distinct from QC radiometry symbols elsewhere in ``bsrn``.
字段规则见 ``specs.LR_SPECS``；磁盘列名（如 ``ghi_avg``）为格式标识。
"""

import numpy as np
import pandas as pd
from .specs import LR_SPECS

# =============================================================================
# TRANSLATION OF: 1_utils.R
# =============================================================================

def get_azimuth_elevation(azimuth=None, elevation=None):
    """Translates getAzimuthElevation"""
    if azimuth is None or elevation is None:
        return "  -1 -1"
    
    az = [float(x) for x in azimuth.split(',')] if isinstance(azimuth, str) else list(azimuth)
    el = [float(x) for x in elevation.split(',')] if isinstance(elevation, str) else list(elevation)
    
    if len(az) != len(el):
        raise ValueError("azimuth and elevation must have same size")
        
    n = len(az)
    pad = 11 - (n % 11) if n % 11 != 0 else 0
    az_padded = az + [-1] * pad
    el_padded = el + [-1] * pad
    
    rows = []
    for i in range(0, len(az_padded), 11):
        line = " ".join([f"{a:>3.0f} {e:>2.0f}" for a, e in zip(az_padded[i:i+11], el_padded[i:i+11])])
        rows.append(f" {line}")
    return "\n".join(rows)

class BSRNRecord:
    """
    Base class replicating the dynamic R6 generic functions.
    Translates: genericInitialize, rw_ActiveBinding, genericIsMandatory, 
    genericIsMissing, genericPrint, etc.
    """
    def __init__(self, lr_code, **kwargs):
        # Bypass custom __setattr__ for initialization
        super().__setattr__('_lr_code', lr_code)
        super().__setattr__('_params', LR_SPECS[lr_code])
        super().__setattr__('_private', {})

        # Set default values
        for var_name, spec in self._params.items():
            self._private[var_name] = spec.get('default')

        # Override defaults with provided kwargs
        for var_name, value in kwargs.items():
            setattr(self, var_name, value)

    def __setattr__(self, name, value):
        """Translates rw_ActiveBinding (Validates on assignment)"""
        if name in ['_lr_code', '_params', '_private']:
            super().__setattr__(name, value)
        elif name in self._params:
            spec = self._params[name]
            val_func_name = spec['validate_func']
            
            if value is not None:
                # Dynamically call the corresponding validation function
                import bsrn.archive.validation as val_module
                val_func = getattr(val_module, val_func_name, lambda x: x)
                try:
                    if val_func_name in (
                        "LR0100_validateFunction",
                        "LR4000_validateFunction",
                    ):
                        ym = self._private.get("yearMonth")
                        clean_val = val_func(value, yearMonth=ym)
                    else:
                        clean_val = val_func(value)
                    clean_val = self._coerce_stored_scalar(name, clean_val)
                except Exception as e:
                    raise ValueError(f"{name}\n {str(e)}")
                self._private[name] = clean_val
            else:
                self._private[name] = None
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._private:
            return self._private[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def is_mandatory(self, var_name):
        """Translates genericIsMandatory"""
        return self._params[var_name]['mandatory']

    def is_missing(self, var_name):
        """Translates genericIsMissing"""
        return self._private[var_name] is None

    def mandatories(self):
        """Translates genericMandatories"""
        return [name for name, spec in self._params.items() if spec['mandatory']]

    def missings(self):
        """Translates genericMissings"""
        return [name for name in self.mandatories() if self.is_missing(name)]

    def is_values_missing(self):
        """Translates genericIsValuesMissing"""
        return len(self.missings()) > 0

    def set_default(self, var_name):
        """Translates genericSetDefault"""
        self._private[var_name] = self._params[var_name]['default']

    def stop_if_values_missing(self, message=""):
        """Translates stopIfValuesMissing"""
        if self.is_values_missing():
            tmp = ", ".join(self.missings())
            raise ValueError(f"{message}\n missing value(s) : {tmp}")

    def _coerce_stored_scalar(self, var_name, value):
        """
        After validation, coerce scalars to Python types matching Fortran ``I`` / ``F`` specs
        (R stores validated integers as integers, not ``33.0``).
        校验后将标量转为与 Fortran ``I``/``F`` 一致的 Python 类型（与 R 一致，避免 ``33.0``）。
        """
        if value is None or isinstance(value, (pd.Series, np.ndarray, list, tuple)):
            return value
        fmt = self._params[var_name].get("format", "")
        if fmt.startswith("I"):
            return int(round(float(value)))
        if fmt.startswith("F"):
            return float(value)
        return value

    def __str__(self):
        """Translates genericPrint"""
        msg = "WARNING : The object is missing value(s).\n" if self.is_values_missing() else ""
        m_vars = self.mandatories()
        for v, spec in self._params.items():
            value = self._private[v]
            if isinstance(value, (list, np.ndarray, pd.Series)) and len(value) > 1:
                value = f"{value[0]} ..."
            req = "[mandatory]" if v in m_vars else "[optional]"
            msg += f"{req} {v} ({spec['label']}) : {value}\n"
        return msg

    def get_format_value(self, var_name):
        """Translates getFormatValue (Fortran padding logic)"""
        value = self._private.get(var_name)
        spec = self._params[var_name]
        missing_code = spec.get('missing')

        if value is None:
            value = missing_code
            if str(value) in ["-999", "-999.9", "-99.9", "-99.99"]:
                value = float(value) if '.' in str(value) else int(value)

        # Vectorized fields: return numeric arrays (formatting for output is ``_format_series_field``).
        # 向量字段：返回数值数组；写出磁盘时的格式见 ``_format_series_field``。
        if isinstance(value, pd.Series):
            return value.fillna(missing_code)
        elif isinstance(value, np.ndarray):
            return np.where(np.isnan(value), missing_code, value)

        fmt = spec['format']
        if value is None: return ""

        if fmt == "L": return "Y" if value else "N"
        if fmt.startswith("I"):
            w = int(fmt[1:])
            return f"{int(value):>{w}d}"
        if fmt.startswith("F"):
            w, d = map(int, fmt[1:].split('.'))
            return f"{float(value):>{w}.{d}f}"
        if fmt.startswith("A"):
            w = int(fmt[1:]) if len(fmt) > 1 else 80
            if fmt == "A": w = 80
            s = str(value)
            return f"{s:<{w}}"[:w]
        return str(value)

    def _format_series_field(self, var_name: str) -> pd.Series:
        """
        One Fortran-formatted string per row for vector LR fields (LR0100 / LR4000).
        Matches scalar ``get_format_value`` padding so outputs are not Python ``str(float)`` (no ``.0`` on integers).
        每分钟一行 Fortran 宽度字符串；与标量 ``get_format_value`` 一致，避免 ``33.0`` 这类输出。
        """
        spec = self._params[var_name]
        miss = spec.get("missing")
        fmt = spec["format"]
        s = pd.Series(self._private[var_name])
        if miss is not None:
            s = s.fillna(miss)
        if fmt.startswith("I"):
            w = int(fmt[1:])
            arr = np.rint(s.to_numpy(dtype=np.float64))
            return pd.Series([f"{int(v):>{w}d}" for v in arr])
        if fmt.startswith("F"):
            fw, fd = fmt[1:].split(".")
            w, d = int(fw), int(fd)
            arr = s.to_numpy(dtype=np.float64)
            return pd.Series([f"{float(v):>{w}.{d}f}" for v in arr])
        raise ValueError(f"Unsupported vector format {fmt!r} for {var_name}")