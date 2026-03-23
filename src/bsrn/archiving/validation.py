"""
BSRN logical-record field validators (R ``1_validateFunc_headers.R`` + ``1_validateFunc_datas.R``).

Each ``*_validateFunction`` name matches ``LR_SPECS`` ``validate_func``; ``BSRNRecord`` loads
the callable by name on field assignment. Code lists come from ``specs`` (``QUANTITIES``, …).

BSRN 逻辑记录字段校验（R ``1_validateFunc_*.R``）。函数名与 ``LR_SPECS`` 中 ``validate_func`` 一致，
由 ``BSRNRecord`` 在赋值时按名调用；分类编码表见 ``specs``。
"""

import calendar
import re
from datetime import datetime

from .specs import QUANTITIES, SURFACES, TOPOGRAPHIES, PYRGEOMETER_BODY, PYRGEOMETER_DOME

# =============================================================================
# TRANSLATION OF: 1_validateFunc_headers.R — header / metadata fields (LR0001–LR0008)
# R 对应 ``1_validateFunc_headers.R``：头记录与元数据字段（LR0001–LR0008）
# =============================================================================

# --- Core Fortran-style checks (I / A / F widths, C numeric, L logical) / 核心 Fortran 风格校验 ---


def I_validateFunction(value, digits, v_min=0, v_max=None):
    """Integer validation function"""
    if v_max is None:
        v_max = (10**digits) - 1
    if not isinstance(value, (int, float)):
        raise ValueError("must be a numerical value")
    if value % 1 != 0:
        raise ValueError("must be an integer")
    if value < v_min or value > v_max:
        raise ValueError(f"must be between {v_min} and {v_max}")
    return int(value)


def C_validateFunction(value):
    """Numerical validation function"""
    if value is None or value == "":
        raise ValueError("Value cannot be NULL or empty")
    return value

def A_validateFunction(value, maxLenght=float('inf')):
    """Character validation function"""
    if not isinstance(value, str):
        raise ValueError("must be a character")
    n = len(value)
    if n < 1 or n > maxLenght:
        raise ValueError(f"number of character must be between 1 and {maxLenght}")
    return value

def F_validateFunction(value, w, d):
    """Fixed point format validation fonction"""
    if not isinstance(value, (int, float)):
        raise ValueError("must be a numerical value")
    s = str(value).split('.')
    if len(s) >= 1:
        # Integer width excludes the minus sign when checking against (w - d). / 校验整数位宽时先去掉负号。
        int_part = s[0].replace('-', '')
        if len(int_part) >= (w - d):
            raise ValueError(f"must be at format F{w}.{d}")
    if len(s) > 1:
        if len(s[1]) > d:
            raise ValueError(f"must be at format F{w}.{d}")
    return float(value)

def L_validateFunction(value):
    """Logical validation function"""
    if not isinstance(value, bool):
        raise ValueError("must be a logical value (TRUE or FALSE)")
    return value


# --- Fixed-width integer tokens (I2 … I8) / 定宽整数（I2 … I8）---


def I2_validateFunction(value):
    """I2 validation function"""
    return I_validateFunction(value, 2)

def I3_validateFunction(value):
    """I3 validation function"""
    return I_validateFunction(value, 3)

def I4_validateFunction(value):
    """I4 validation function"""
    return I_validateFunction(value, 4)

def I5_validateFunction(value):
    """I5 validation function"""
    return I_validateFunction(value, 5)

def I8_validateFunction(value):
    """I8 validation function"""
    return I_validateFunction(value, 8)


# --- Calendar parts (month, year, day, hour, minute) / 日期时间分量 ---


def month_validateFunction(value):
    """month validation function"""
    return I_validateFunction(value, 2, 1, 12)

def year_validateFunction(value):
    """year validation function"""
    return I_validateFunction(value, 4, 1992)

def day_validateFunction(value):
    """day validation function"""
    return I_validateFunction(value, 2, 1, 31)

def hour_validateFunction(value):
    """hour validation function"""
    return I_validateFunction(value, 2, 0, 23)

def minute_validateFunction(value):
    """minute validation function"""
    return I_validateFunction(value, 2, 0, 59)


# --- Fixed-width strings (A5 … A80) / 定宽字符串（A5 … A80）---


def A5_validateFunction(value):
    """A5 validation function"""
    return A_validateFunction(value, 5)

def A15_validateFunction(value):
    """A15 validation function"""
    return A_validateFunction(value, 15)

def A18_validateFunction(value):
    """A18 validation function"""
    return A_validateFunction(value, 18)

def A25_validateFunction(value):
    """A25 validation function"""
    return A_validateFunction(value, 25)

def A30_validateFunction(value):
    """A30 validation function"""
    return A_validateFunction(value, 30)

def A38_validateFunction(value):
    """A38 validation function"""
    return A_validateFunction(value, 38)

def A40_validateFunction(value):
    """A40 validation function"""
    return A_validateFunction(value, 40)

def A80_validateFunction(value):
    """A80 validation function"""
    return A_validateFunction(value, 80)


# --- Contact / network string patterns (telephone, TCP/IP, e-mail) / 联系方式与网络格式 ---


def telephone_validateFunction(value):
    """telephone validation function"""
    A_validateFunction(value, 20)
    telephoneRegex = r"^[+]?[\d\s]{8,20}$"
    if not re.search(telephoneRegex, value):
        raise ValueError("must have a telephone format")
    return value

def tcpip_validateFunction(value):
    """tcpip validation function"""
    A_validateFunction(value, 15)
    tcpipRegex = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    if not re.search(tcpipRegex, value):
        raise ValueError("must have a TCP/IP format")
    return value

def email_validateFunction(value):
    """email validation function"""
    A_validateFunction(value, 50)
    emailRegex = r"^[\w\.-]+@[\w-]+\.[\w]{2,4}$"
    if not re.search(emailRegex, value):
        raise ValueError("must have a e-mail format")
    return value


# --- Encoded fields vs ``specs`` tables (A3–A7) / 与 ``specs`` 编码表对照的字段 ---


def quantities_validateFunction(value):
    """radiation quantities measured validation function"""
    if value not in QUANTITIES.values():
        raise ValueError("must be in A5 quantities (specs.QUANTITIES)")
    return value

def surface_validateFunction(value):
    """surface validation function"""
    if value not in SURFACES.values():
        raise ValueError("must be in A4 surfaces (specs.SURFACES)")
    return value

def topography_validateFunction(value):
    """topography type validation function"""
    if value not in TOPOGRAPHIES.values():
        raise ValueError("must be in A3 topographies (specs.TOPOGRAPHIES)")
    return value

def body_validateFunction(value):
    """pyrgeometer body validation function"""
    if value not in PYRGEOMETER_BODY.values():
        raise ValueError("must be in A6 pyrgeometer body (specs.PYRGEOMETER_BODY)")
    return value

def dome_validateFunction(value):
    """pyrgeometer dome validation function"""
    if value not in PYRGEOMETER_DOME.values():
        raise ValueError("must be in A7 pyrgeometer dome (specs.PYRGEOMETER_DOME)")
    return value


# --- Float formats F7.3 and F12.4 (underscores replace dots in Python names) / 浮点格式 ---


def F7_3_validateFunction(value):
    """F7.3 validation function (Renamed to avoid dot in Python func name)"""
    return F_validateFunction(value, 7, 3)

def F12_4_validateFunction(value):
    """F12.4 validation function (Renamed to avoid dot in Python func name)"""
    return F_validateFunction(value, 12, 4)


# --- Lat/lon (F7.3), zenith (I2), horizon comma lists, MM/DD/YY in A8 / 经纬度、天顶角、地平线、A8 日期串 ---


def latitude_validateFunction(value):
    """latitude validation function"""
    F_validateFunction(value, 7, 3)
    latitudeRegex = r"^(?:1[0-7][0-9]|[0-9]?[0-9])\.[0-9]{3}$"
    if not re.search(latitudeRegex, str(value)):
        raise ValueError("must have a latitude format")
    return value

def longitude_validateFunction(value):
    """longitude validation function"""
    F_validateFunction(value, 7, 3)
    longitudeRegex = r"^(?:3[0-5][0-9]|2[0-9][0-9]|[01]?[0-9]?[0-9])\.[0-9]{3}$"
    if not re.search(longitudeRegex, str(value)):
        raise ValueError("must have a longitude format")
    return value


def date_validateFunction(value):
    """date validation function"""
    A_validateFunction(value, 8)
    try:
        datetime.strptime(value, '%m/%d/%y')
    except ValueError:
        raise ValueError('must have format "MM/DD/YY"')
    return value

def zenith_validateFunction(value):
    """zenith validation function"""
    return I_validateFunction(value, 2, 0, 90)

def azimuth_validateFunction(value):
    """azimuth validation function"""
    A_validateFunction(value)
    azimuthRegex = r"^(?:(?:3[0-5][0-9]|2[0-9][0-9]|[01]?[0-9]?[0-9]),)*(?:(?:3[0-5][0-9]|2[0-9][0-9]|[01]?[0-9]?[0-9])){1}$"
    if not re.search(azimuthRegex, str(value)):
        raise ValueError("must have a azimuth format (A1,A2,...,An)")
    return value

def elevation_validateFunction(value):
    """elevation validation function"""
    A_validateFunction(value)
    elevationRegex = r"^(?:(?:[0-8]?[0-9]),)*(?:(?:[0-8]?[0-9])){1}$"
    if not re.search(elevationRegex, str(value)):
        raise ValueError("must have a elevation format (E1,E2,...,En)")
    return value


# =============================================================================
# TRANSLATION OF: 1_validateFunc_datas.R — LR0100 / LR4000 minute-series fields
# R 对应 ``1_validateFunc_datas.R``：LR0100 / LR4000 分钟序列字段
# =============================================================================

# ``yearMonth`` token (seven chars ``YYYY-MM``). / ``yearMonth`` 七字符 ``YYYY-MM``。
_YEAR_MONTH_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})$")


def genericValidateFunction(value):
    """
    Validate ``yearMonth`` tokens (format A7 in LR0100 / LR4000: ``'YYYY-MM'``).
    校验 LR0100 / LR4000 中的 ``yearMonth``（规范格式 A7：``'YYYY-MM'``）。
    """
    if not isinstance(value, str):
        raise ValueError("must be a character string")
    m = _YEAR_MONTH_RE.match(value)
    if not m:
        raise ValueError("must match 'YYYY-MM' (7 characters)")
    month = int(m.group("m"))
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12")
    return value


def LR0100_validateFunction(value, yearMonth=None):
    """
    Vector length must match ``numberOfMinutes(yearMonth)`` when ``yearMonth`` is set.
    当 ``yearMonth`` 已设置时，向量长度须等于 ``numberOfMinutes(yearMonth)``。
    """
    if yearMonth is None:
        return value
    y, mo = map(int, yearMonth.split("-"))
    n = calendar.monthrange(y, mo)[1] * 1440
    if len(value) != n:
        raise ValueError(f"The size of vector must be {n}")
    return value


def LR4000_validateFunction(value, yearMonth=None):
    """
    Vector length must match ``numberOfMinutes(yearMonth)`` when ``yearMonth`` is set.
    当 ``yearMonth`` 已设置时，向量长度须等于 ``numberOfMinutes(yearMonth)``。
    """
    if yearMonth is None:
        return value
    y, mo = map(int, yearMonth.split("-"))
    n = calendar.monthrange(y, mo)[1] * 1440
    if len(value) != n:
        raise ValueError(f"The size of vector must be {n}")
    return value