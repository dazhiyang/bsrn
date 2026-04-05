"""
BSRN logical-record field validators (R ``1_validateFunc_headers.R`` + ``1_validateFunc_datas.R``).

Each ``*_validateFunction`` name matches ``LR_SPECS`` ``validate_func``; Pydantic
``ArchiveRecordBase`` runs them in a model validator. Code lists come from ``specs``
(``QUANTITIES``, …).

BSRN 逻辑记录字段校验（R ``1_validateFunc_*.R``）。函数名与 ``LR_SPECS`` 中 ``validate_func`` 一致，
由 Pydantic ``ArchiveRecordBase`` 的模型校验器按名调用；分类编码表见 ``specs``。
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
    """
    Validate integer ``value`` for Fortran ``I`` width and optional bounds.

    Translates from R integer width checks (``1_validateFunc_headers.R``).
    对应 R 整数宽度校验（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate integer.
        待校验整数。
    digits : int
        Number of decimal digits allowed (``10**digits - 1`` upper bound if ``v_max`` is None).
        十进制位数（``v_max`` 为 None 时上界为 ``10**digits - 1``）。
    v_min : int, optional
        Minimum inclusive value; default ``0``.
        最小值（含）；默认 ``0``。
    v_max : int or None, optional
        Maximum inclusive value; if ``None``, set from ``digits``.
        最大值（含）；``None`` 时由 ``digits`` 推导。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        If ``value`` is not integral or outside ``[v_min, v_max]``.
        非整数或超出 ``[v_min, v_max]`` 时。
    """
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
    """
    Accept any non-empty numeric-like token (R ``C`` / free numeric).

    Translates from R ``C`` validation (``1_validateFunc_headers.R``).
    对应 R ``C`` 校验（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : object
        Candidate value.
        待校验值。

    Returns
    -------
    object
        ``value`` unchanged when valid.
        合法时原样返回。

    Raises
    ------
    ValueError
        If ``value`` is ``None`` or empty string.
        ``value`` 为 ``None`` 或空字符串时。
    """
    if value is None or value == "":
        raise ValueError("Value cannot be NULL or empty")
    return value


def A_validateFunction(value, maxLenght=float("inf")):
    """
    Validate non-empty string length for Fortran ``A`` fields.

    Translates from R ``A`` validation (``1_validateFunc_headers.R``).
    对应 R ``A`` 校验（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。
    maxLenght : float, optional
        Maximum length (default: no upper bound beyond positivity).
        最大长度（默认：仅要求正长度）。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        If not a ``str`` or length outside ``[1, maxLenght]``.
        非字符串或长度不在 ``[1, maxLenght]`` 时。
    """
    if not isinstance(value, str):
        raise ValueError("must be a character")
    n = len(value)
    if n < 1 or n > maxLenght:
        raise ValueError(f"number of character must be between 1 and {maxLenght}")
    return value


def F_validateFunction(value, w, d):
    """
    Validate float width for Fortran ``F{w}.{d}`` (integer and fractional parts).

    Translates from R ``F`` validation (``1_validateFunc_headers.R``).
    对应 R ``F`` 校验（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate numeric.
        待校验数值。
    w : int
        Total field width.
        总字段宽度。
    d : int
        Digits after the decimal point.
        小数位数。

    Returns
    -------
    float
        Validated float.
        校验后的浮点数。

    Raises
    ------
    ValueError
        If not numeric or if string representation violates ``F{w}.{d}``.
        非数值或字符串形式不符合 ``F{w}.{d}`` 时。
    """
    if not isinstance(value, (int, float)):
        raise ValueError("must be a numerical value")
    s = str(value).split(".")
    if len(s) >= 1:
        # Integer width excludes the minus sign when checking against (w - d). / 校验整数位宽时先去掉负号。
        int_part = s[0].replace("-", "")
        if len(int_part) >= (w - d):
            raise ValueError(f"must be at format F{w}.{d}")
    if len(s) > 1:
        if len(s[1]) > d:
            raise ValueError(f"must be at format F{w}.{d}")
    return float(value)


def L_validateFunction(value):
    """
    Validate Python ``bool`` for Fortran logical ``L`` fields.

    Translates from R ``L`` validation (``1_validateFunc_headers.R``).
    对应 R ``L`` 校验（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : bool
        Candidate flag.
        待校验布尔值。

    Returns
    -------
    bool
        Validated boolean.
        校验后的布尔值。

    Raises
    ------
    ValueError
        If ``value`` is not a ``bool``.
        ``value`` 非 ``bool`` 时。
    """
    if not isinstance(value, bool):
        raise ValueError("must be a logical value (TRUE or FALSE)")
    return value


# --- Fixed-width integer tokens (I2 … I8) / 定宽整数（I2 … I8）---


def I2_validateFunction(value):
    """
    Validate Fortran ``I2`` integer field.

    Translates from R ``I2`` token rule (``1_validateFunc_headers.R``).
    对应 R ``I2`` 规则（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 2)


def I3_validateFunction(value):
    """
    Validate Fortran ``I3`` integer field.

    Translates from R ``I3`` token rule (``1_validateFunc_headers.R``).
    对应 R ``I3`` 规则（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 3)


def I4_validateFunction(value):
    """
    Validate Fortran ``I4`` integer field.

    Translates from R ``I4`` token rule (``1_validateFunc_headers.R``).
    对应 R ``I4`` 规则（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 4)


def I5_validateFunction(value):
    """
    Validate Fortran ``I5`` integer field.

    Translates from R ``I5`` token rule (``1_validateFunc_headers.R``).
    对应 R ``I5`` 规则（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 5)


def I8_validateFunction(value):
    """
    Validate Fortran ``I8`` integer field.

    Translates from R ``I8`` token rule (``1_validateFunc_headers.R``).
    对应 R ``I8`` 规则（``1_validateFunc_headers.R``）。

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 8)


# --- Calendar parts (month, year, day, hour, minute) / 日期时间分量 ---


def month_validateFunction(value):
    """
    Validate calendar month in ``1``–``12`` (``I2``).

    Parameters
    ----------
    value : int or float
        Candidate month.
        待校验月份。

    Returns
    -------
    int
        Validated month.
        校验后的月份。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 2, 1, 12)


def year_validateFunction(value):
    """
    Validate four-digit year (``I4``, minimum ``1992`` per BSRN spec).

    Parameters
    ----------
    value : int or float
        Candidate year.
        待校验年份。

    Returns
    -------
    int
        Validated year.
        校验后的年份。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 4, 1992)


def day_validateFunction(value):
    """
    Validate day-of-month in ``1``–``31`` (``I2``).

    Parameters
    ----------
    value : int or float
        Candidate day.
        待校验日。

    Returns
    -------
    int
        Validated day.
        校验后的日。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 2, 1, 31)


def hour_validateFunction(value):
    """
    Validate hour of day ``0``–``23`` (``I2``).

    Parameters
    ----------
    value : int or float
        Candidate hour.
        待校验小时。

    Returns
    -------
    int
        Validated hour.
        校验后的小时。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 2, 0, 23)


def minute_validateFunction(value):
    """
    Validate minute ``0``–``59`` (``I2``).

    Parameters
    ----------
    value : int or float
        Candidate minute.
        待校验分钟。

    Returns
    -------
    int
        Validated minute.
        校验后的分钟。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 2, 0, 59)


# --- Fixed-width strings (A5 … A80) / 定宽字符串（A5 … A80）---


def A5_validateFunction(value):
    """
    Validate Fortran ``A5`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 5)


def A15_validateFunction(value):
    """
    Validate Fortran ``A15`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 15)


def A18_validateFunction(value):
    """
    Validate Fortran ``A18`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 18)


def A25_validateFunction(value):
    """
    Validate Fortran ``A25`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 25)


def A30_validateFunction(value):
    """
    Validate Fortran ``A30`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 30)


def A38_validateFunction(value):
    """
    Validate Fortran ``A38`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 38)


def A40_validateFunction(value):
    """
    Validate Fortran ``A40`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 40)


def A80_validateFunction(value):
    """
    Validate Fortran ``A80`` string field.

    Parameters
    ----------
    value : str
        Candidate string.
        待校验字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction`.
        由 :func:`A_validateFunction` 抛出。
    """
    return A_validateFunction(value, 80)


# --- Contact / network string patterns (telephone, TCP/IP, e-mail) / 联系方式与网络格式 ---


def telephone_validateFunction(value):
    """
    Validate telephone string (length + simple digit pattern).

    Parameters
    ----------
    value : str
        Candidate telephone text.
        待校验电话字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction` or if pattern does not match.
        :func:`A_validateFunction` 失败或正则不匹配时。
    """
    A_validateFunction(value, 20)
    telephoneRegex = r"^[+]?[\d\s]{8,20}$"
    if not re.search(telephoneRegex, value):
        raise ValueError("must have a telephone format")
    return value


def tcpip_validateFunction(value):
    """
    Validate IPv4-style TCP/IP string (``A15``).

    Parameters
    ----------
    value : str
        Candidate address.
        待校验地址。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction` or if pattern does not match.
        :func:`A_validateFunction` 失败或正则不匹配时。
    """
    A_validateFunction(value, 15)
    tcpipRegex = (
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )
    if not re.search(tcpipRegex, value):
        raise ValueError("must have a TCP/IP format")
    return value


def email_validateFunction(value):
    """
    Validate e-mail string (``A50``).

    Parameters
    ----------
    value : str
        Candidate e-mail.
        待校验邮箱。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction` or if pattern does not match.
        :func:`A_validateFunction` 失败或正则不匹配时。
    """
    A_validateFunction(value, 50)
    emailRegex = r"^[\w\.-]+@[\w-]+\.[\w]{2,4}$"
    if not re.search(emailRegex, value):
        raise ValueError("must have a e-mail format")
    return value


# --- Encoded fields vs ``specs`` tables (A3–A7) / 与 ``specs`` 编码表对照的字段 ---


def quantities_validateFunction(value):
    """
    Validate radiation quantity code against ``specs.QUANTITIES``.

    Parameters
    ----------
    value : object
        Candidate code (must match a value in ``QUANTITIES``).
        待校验编码（须为 ``QUANTITIES`` 中取值之一）。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        If ``value`` is not in ``QUANTITIES.values()``.
        ``value`` 不在 ``QUANTITIES.values()`` 中时。
    """
    if value not in QUANTITIES.values():
        raise ValueError("must be in A5 quantities (specs.QUANTITIES)")
    return value


def surface_validateFunction(value):
    """
    Validate surface type code against ``specs.SURFACES``.

    Parameters
    ----------
    value : object
        Candidate code.
        待校验编码。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        If ``value`` is not in ``SURFACES.values()``.
        ``value`` 不在 ``SURFACES.values()`` 中时。
    """
    if value not in SURFACES.values():
        raise ValueError("must be in A4 surfaces (specs.SURFACES)")
    return value


def topography_validateFunction(value):
    """
    Validate topography code against ``specs.TOPOGRAPHIES``.

    Parameters
    ----------
    value : object
        Candidate code.
        待校验编码。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        If ``value`` is not in ``TOPOGRAPHIES.values()``.
        ``value`` 不在 ``TOPOGRAPHIES.values()`` 中时。
    """
    if value not in TOPOGRAPHIES.values():
        raise ValueError("must be in A3 topographies (specs.TOPOGRAPHIES)")
    return value


def body_validateFunction(value):
    """
    Validate pyrgeometer body compensation code against ``specs.PYRGEOMETER_BODY``.

    Parameters
    ----------
    value : object
        Candidate code.
        待校验编码。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        If ``value`` is not in ``PYRGEOMETER_BODY.values()``.
        ``value`` 不在 ``PYRGEOMETER_BODY.values()`` 中时。
    """
    if value not in PYRGEOMETER_BODY.values():
        raise ValueError("must be in A6 pyrgeometer body (specs.PYRGEOMETER_BODY)")
    return value


def dome_validateFunction(value):
    """
    Validate pyrgeometer dome compensation code against ``specs.PYRGEOMETER_DOME``.

    Parameters
    ----------
    value : object
        Candidate code.
        待校验编码。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        If ``value`` is not in ``PYRGEOMETER_DOME.values()``.
        ``value`` 不在 ``PYRGEOMETER_DOME.values()`` 中时。
    """
    if value not in PYRGEOMETER_DOME.values():
        raise ValueError("must be in A7 pyrgeometer dome (specs.PYRGEOMETER_DOME)")
    return value


# --- Float formats F7.3 and F12.4 (underscores replace dots in Python names) / 浮点格式 ---


def F7_3_validateFunction(value):
    """
    Validate ``F7.3`` float field (underscore name avoids a dot in Python).

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    float
        Validated float.
        校验后的浮点数。

    Raises
    ------
    ValueError
        From :func:`F_validateFunction`.
        由 :func:`F_validateFunction` 抛出。
    """
    return F_validateFunction(value, 7, 3)


def F12_4_validateFunction(value):
    """
    Validate ``F12.4`` float field (underscore name avoids a dot in Python).

    Parameters
    ----------
    value : int or float
        Candidate value.
        待校验值。

    Returns
    -------
    float
        Validated float.
        校验后的浮点数。

    Raises
    ------
    ValueError
        From :func:`F_validateFunction`.
        由 :func:`F_validateFunction` 抛出。
    """
    return F_validateFunction(value, 12, 4)


# --- Lat/lon (F7.3), zenith (I2), horizon comma lists, MM/DD/YY in A8 / 经纬度、天顶角、地平线、A8 日期串 ---


def latitude_validateFunction(value):
    """
    Validate latitude string matching ``F7.3`` and regex pattern.

    Parameters
    ----------
    value : int or float or str
        Candidate latitude (degrees).
        待校验纬度（度）。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        From :func:`F_validateFunction` or if regex fails.
        :func:`F_validateFunction` 失败或正则不匹配时。
    """
    F_validateFunction(value, 7, 3)
    latitudeRegex = r"^(?:1[0-7][0-9]|[0-9]?[0-9])\.[0-9]{3}$"
    if not re.search(latitudeRegex, str(value)):
        raise ValueError("must have a latitude format")
    return value


def longitude_validateFunction(value):
    """
    Validate longitude string matching ``F7.3`` and regex pattern.

    Parameters
    ----------
    value : int or float or str
        Candidate longitude (degrees).
        待校验经度（度）。

    Returns
    -------
    object
        ``value`` when valid.
        合法时返回 ``value``。

    Raises
    ------
    ValueError
        From :func:`F_validateFunction` or if regex fails.
        :func:`F_validateFunction` 失败或正则不匹配时。
    """
    F_validateFunction(value, 7, 3)
    longitudeRegex = r"^(?:3[0-5][0-9]|2[0-9][0-9]|[01]?[0-9]?[0-9])\.[0-9]{3}$"
    if not re.search(longitudeRegex, str(value)):
        raise ValueError("must have a longitude format")
    return value


def date_validateFunction(value):
    """
    Validate ``MM/DD/YY`` date string (``A8``).

    Parameters
    ----------
    value : str
        Candidate date text.
        待校验日期字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction` or if ``strptime`` fails.
        :func:`A_validateFunction` 失败或日期解析失败时。
    """
    A_validateFunction(value, 8)
    try:
        datetime.strptime(value, "%m/%d/%y")
    except ValueError:
        raise ValueError('must have format "MM/DD/YY"')
    return value


def zenith_validateFunction(value):
    """
    Validate zenith angle ``0``–``90`` degrees (``I2``).

    Parameters
    ----------
    value : int or float
        Candidate zenith angle.
        待校验天顶角。

    Returns
    -------
    int
        Validated integer.
        校验后的整数。

    Raises
    ------
    ValueError
        From :func:`I_validateFunction`.
        由 :func:`I_validateFunction` 抛出。
    """
    return I_validateFunction(value, 2, 0, 90)


def azimuth_validateFunction(value):
    """
    Validate comma-separated azimuth list ``A1,A2,...`` (free ``A`` width).

    Parameters
    ----------
    value : str
        Candidate azimuth list string.
        待校验方位角列表字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction` or if regex fails.
        :func:`A_validateFunction` 失败或正则不匹配时。
    """
    A_validateFunction(value)
    azimuthRegex = (
        r"^(?:(?:3[0-5][0-9]|2[0-9][0-9]|[01]?[0-9]?[0-9]),)*"
        r"(?:(?:3[0-5][0-9]|2[0-9][0-9]|[01]?[0-9]?[0-9])){1}$"
    )
    if not re.search(azimuthRegex, str(value)):
        raise ValueError("must have a azimuth format (A1,A2,...,An)")
    return value


def elevation_validateFunction(value):
    """
    Validate comma-separated horizon elevation list ``E1,E2,...``.

    Parameters
    ----------
    value : str
        Candidate elevation list string.
        待校验高度角列表字符串。

    Returns
    -------
    str
        Validated string.
        校验后的字符串。

    Raises
    ------
    ValueError
        From :func:`A_validateFunction` or if regex fails.
        :func:`A_validateFunction` 失败或正则不匹配时。
    """
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
    Validate ``yearMonth`` token (``A7`` in LR0100 / LR4000: ``'YYYY-MM'``).

    Translates from R ``genericValidateFunction`` (``1_validateFunc_datas.R``).
    对应 R ``genericValidateFunction``（``1_validateFunc_datas.R``）。

    Parameters
    ----------
    value : str
        Seven-character ``'YYYY-MM'`` string.
        七字符 ``'YYYY-MM'`` 字符串。

    Returns
    -------
    str
        The same token when valid.
        合法时原样返回。

    Raises
    ------
    ValueError
        If not a string, wrong pattern, or month outside ``1``–``12``.
        非字符串、格式不符或月份越界时。
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
    Ensure minute-series length matches the month when ``yearMonth`` is set.

    Translates from R ``LR0100_validateFunction`` (``1_validateFunc_datas.R``).
    对应 R ``LR0100_validateFunction``（``1_validateFunc_datas.R``）。

    Parameters
    ----------
    value : sequence
        Per-minute vector (e.g. ``list``, ``numpy.ndarray``, ``pandas.Series``).
        每分钟一个元素的向量（如 ``list``、``numpy.ndarray``、``pandas.Series``）。
    yearMonth : str or None, optional
        ``'YYYY-MM'`` token; if ``None``, skip length check.
        ``'YYYY-MM'``；为 ``None`` 时不检查长度。

    Returns
    -------
    sequence
        ``value`` unchanged when valid or when ``yearMonth`` is ``None``.
        合法或 ``yearMonth`` 为 ``None`` 时原样返回。

    Raises
    ------
    ValueError
        If ``len(value)`` does not equal days in month × 1440.
        ``len(value)`` 不等于当月天数 × 1440 时。
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
    Ensure LR4000 minute-series length matches the month when ``yearMonth`` is set.

    Translates from R ``LR4000_validateFunction`` (``1_validateFunc_datas.R``).
    对应 R ``LR4000_validateFunction``（``1_validateFunc_datas.R``）。

    Parameters
    ----------
    value : sequence
        Per-minute vector (e.g. ``list``, ``numpy.ndarray``, ``pandas.Series``).
        每分钟一个元素的向量。
    yearMonth : str or None, optional
        ``'YYYY-MM'`` token; if ``None``, skip length check.
        ``'YYYY-MM'``；为 ``None`` 时不检查长度。

    Returns
    -------
    sequence
        ``value`` unchanged when valid or when ``yearMonth`` is ``None``.
        合法或 ``yearMonth`` 为 ``None`` 时原样返回。

    Raises
    ------
    ValueError
        If ``len(value)`` does not equal days in month × 1440.
        ``len(value)`` 不等于当月天数 × 1440 时。
    """
    if yearMonth is None:
        return value
    y, mo = map(int, yearMonth.split("-"))
    n = calendar.monthrange(y, mo)[1] * 1440
    if len(value) != n:
        raise ValueError(f"The size of vector must be {n}")
    return value