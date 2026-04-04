"""
Explicit Pydantic logical-record models aligned with ``specs.LR_SPECS``.

Each scalar/header field uses :func:`lr_spec` once: it attaches both BSRN
``json_schema_extra`` (Fortran layout) and post-parse validation
(``Annotated`` + ``AfterValidator``). LR0100 / LR4000 minute columns use
``lr_spec_field`` plus a shared ``field_validator`` that reads ``yearMonth``.

显式 Pydantic 逻辑记录模型。**标量/头字段** 用 ``lr_spec`` 一次完成元数据 + 校验；
**LR0100/LR4000 分钟列** 用 ``lr_spec_field`` 与 ``field_validator``（依赖 ``yearMonth``）。
"""

from __future__ import annotations

from typing import Annotated, Optional, Union

import numpy as np
import pandas as pd
from pydantic import AfterValidator, ConfigDict, Field, ValidationInfo, field_validator

from .archive_lr_formats import _FORMATTERS
from .records_base import ArchiveRecordBase, _validation_callable, make_archive_after_validator
from .specs import LR_SPECS

ConstScalar = Union[str, int, float]
MinuteVector = Optional[Union[pd.Series, np.ndarray]]

_LR0100_MINUTE_FIELDS = tuple(k for k in LR_SPECS["LR0100"] if k != "yearMonth")
_LR4000_MINUTE_FIELDS = tuple(k for k in LR_SPECS["LR4000"] if k != "yearMonth")


def lr_spec_field(lr: str, fname: str, **kwargs):
    """Pydantic ``Field`` with ``LR_SPECS`` archive metadata (minute columns, Fortran layout)."""
    return Field(json_schema_extra={"archive": dict(LR_SPECS[lr][fname])}, **kwargs)


def lr_spec(lr: str, fname: str, value_type, **field_kwargs):
    """One logical-record field: ``Annotated`` + ``AfterValidator`` + ``Field`` from ``LR_SPECS``."""
    meta = dict(LR_SPECS[lr][fname])
    return Annotated[
        value_type,
        AfterValidator(make_archive_after_validator(lr, fname)),
        Field(json_schema_extra={"archive": meta}, **field_kwargs),
    ]


class LR0001(ArchiveRecordBase):
    stationNumber: lr_spec("LR0001", "stationNumber", int)
    month: lr_spec("LR0001", "month", int)
    year: lr_spec("LR0001", "year", int)
    version: lr_spec("LR0001", "version", int)


class LR0002(ArchiveRecordBase):
    scientistChange: lr_spec("LR0002", "scientistChange", bool, default=False)
    scientistChangeDay: lr_spec("LR0002", "scientistChangeDay", Optional[int], default=None)
    scientistChangeHour: lr_spec("LR0002", "scientistChangeHour", Optional[int], default=None)
    scientistChangeMinute: lr_spec("LR0002", "scientistChangeMinute", Optional[int], default=None)
    scientistName: lr_spec("LR0002", "scientistName", str)
    scientistTel: lr_spec("LR0002", "scientistTel", str)
    scientistFax: lr_spec("LR0002", "scientistFax", str)
    scientistTcpip: lr_spec("LR0002", "scientistTcpip", Optional[str], default=None)
    scientistMail: lr_spec("LR0002", "scientistMail", Optional[str], default=None)
    scientistAddress: lr_spec("LR0002", "scientistAddress", str)
    deputyChange: lr_spec("LR0002", "deputyChange", bool, default=False)
    deputyChangeDay: lr_spec("LR0002", "deputyChangeDay", Optional[int], default=None)
    deputyChangeHour: lr_spec("LR0002", "deputyChangeHour", Optional[int], default=None)
    deputyChangeMinute: lr_spec("LR0002", "deputyChangeMinute", Optional[int], default=None)
    deputyName: lr_spec("LR0002", "deputyName", str)
    deputyTel: lr_spec("LR0002", "deputyTel", str)
    deputyFax: lr_spec("LR0002", "deputyFax", str)
    deputyTcpip: lr_spec("LR0002", "deputyTcpip", Optional[str], default=None)
    deputyMail: lr_spec("LR0002", "deputyMail", Optional[str], default=None)
    deputyAddress: lr_spec("LR0002", "deputyAddress", str)


class LR0003(ArchiveRecordBase):
    message: lr_spec("LR0003", "message", Optional[str], default=None)


class LR0004(ArchiveRecordBase):
    stationDescChange: lr_spec("LR0004", "stationDescChange", bool, default=False)
    stationDescChangeDay: lr_spec("LR0004", "stationDescChangeDay", Optional[int], default=None)
    stationDescChangeHour: lr_spec("LR0004", "stationDescChangeHour", Optional[int], default=None)
    stationDescChangeMinute: lr_spec("LR0004", "stationDescChangeMinute", Optional[int], default=None)
    surfaceType: lr_spec("LR0004", "surfaceType", int)
    topographyType: lr_spec("LR0004", "topographyType", int)
    address: lr_spec("LR0004", "address", str)
    telephone: lr_spec("LR0004", "telephone", Optional[str], default=None)
    fax: lr_spec("LR0004", "fax", Optional[str], default=None)
    tcpip: lr_spec("LR0004", "tcpip", Optional[str], default=None)
    mail: lr_spec("LR0004", "mail", Optional[str], default=None)
    latitude: lr_spec("LR0004", "latitude", float)
    longitude: lr_spec("LR0004", "longitude", float)
    altitude: lr_spec("LR0004", "altitude", int)
    synop: lr_spec("LR0004", "synop", Optional[str], default=None)
    horizonChange: lr_spec("LR0004", "horizonChange", bool, default=False)
    horizonChangeDay: lr_spec("LR0004", "horizonChangeDay", Optional[int], default=None)
    horizonChangeHour: lr_spec("LR0004", "horizonChangeHour", Optional[int], default=None)
    horizonChangeMinute: lr_spec("LR0004", "horizonChangeMinute", Optional[int], default=None)
    azimuth: lr_spec("LR0004", "azimuth", Optional[str], default=None)
    elevation: lr_spec("LR0004", "elevation", Optional[str], default=None)


class LR0005(ArchiveRecordBase):
    change: lr_spec("LR0005", "change", bool, default=False)
    changeDay: lr_spec("LR0005", "changeDay", Optional[int], default=None)
    changeHour: lr_spec("LR0005", "changeHour", Optional[int], default=None)
    changeMinute: lr_spec("LR0005", "changeMinute", Optional[int], default=None)
    operating: lr_spec("LR0005", "operating", bool, default=False)
    manufacturer: lr_spec("LR0005", "manufacturer", str)
    location: lr_spec("LR0005", "location", str)
    distanceFromSite: lr_spec("LR0005", "distanceFromSite", int)
    time1stLaunch: lr_spec("LR0005", "time1stLaunch", Optional[int], default=None)
    time2ndLaunch: lr_spec("LR0005", "time2ndLaunch", Optional[int], default=None)
    time3rdLaunch: lr_spec("LR0005", "time3rdLaunch", Optional[int], default=None)
    time4thLaunch: lr_spec("LR0005", "time4thLaunch", Optional[int], default=None)
    identification: lr_spec("LR0005", "identification", str)
    remarks: lr_spec("LR0005", "remarks", Optional[str], default=None)


class LR0006(ArchiveRecordBase):
    change: lr_spec("LR0006", "change", bool, default=False)
    changeDay: lr_spec("LR0006", "changeDay", Optional[int], default=None)
    changeHour: lr_spec("LR0006", "changeHour", Optional[int], default=None)
    changeMinute: lr_spec("LR0006", "changeMinute", Optional[int], default=None)
    operating: lr_spec("LR0006", "operating", bool, default=False)
    manufacturer: lr_spec("LR0006", "manufacturer", str)
    location: lr_spec("LR0006", "location", str)
    distanceFromSite: lr_spec("LR0006", "distanceFromSite", int)
    identification: lr_spec("LR0006", "identification", str)
    remarks: lr_spec("LR0006", "remarks", Optional[str], default=None)


class LR0007(ArchiveRecordBase):
    change: lr_spec("LR0007", "change", bool, default=False)
    changeDay: lr_spec("LR0007", "changeDay", Optional[int], default=None)
    changeHour: lr_spec("LR0007", "changeHour", Optional[int], default=None)
    changeMinute: lr_spec("LR0007", "changeMinute", Optional[int], default=None)
    cloudAmount: lr_spec("LR0007", "cloudAmount", Optional[str], default=None)
    cloudBaseHeight: lr_spec("LR0007", "cloudBaseHeight", Optional[str], default=None)
    cloudLiquid: lr_spec("LR0007", "cloudLiquid", Optional[str], default=None)
    cloudAerosol: lr_spec("LR0007", "cloudAerosol", Optional[str], default=None)
    waterVapour: lr_spec("LR0007", "waterVapour", Optional[str], default=None)


class LR0008(ArchiveRecordBase):
    change: lr_spec("LR0008", "change", bool, default=False)
    changeDay: lr_spec("LR0008", "changeDay", Optional[int], default=None)
    changeHour: lr_spec("LR0008", "changeHour", Optional[int], default=None)
    changeMinute: lr_spec("LR0008", "changeMinute", Optional[int], default=None)
    operating: lr_spec("LR0008", "operating", bool, default=False)
    radiationQuantityMeasured: lr_spec("LR0008", "radiationQuantityMeasured", int)
    manufacturer: lr_spec("LR0008", "manufacturer", str)
    model: lr_spec("LR0008", "model", str)
    serialNumber: lr_spec("LR0008", "serialNumber", str)
    dateOfPurchase: lr_spec("LR0008", "dateOfPurchase", Optional[str], default=None)
    identification: lr_spec("LR0008", "identification", int)
    remarks: lr_spec("LR0008", "remarks", Optional[str], default=None)
    pyrgeometerBody: lr_spec("LR0008", "pyrgeometerBody", Optional[int], default=None)
    pyrgeometerDome: lr_spec("LR0008", "pyrgeometerDome", Optional[int], default=None)
    numOfBand: lr_spec("LR0008", "numOfBand", Optional[int], default=None)
    wavelenghBand1: lr_spec("LR0008", "wavelenghBand1", Optional[float], default=None)
    bandwidthBand1: lr_spec("LR0008", "bandwidthBand1", Optional[float], default=None)
    wavelenghBand2: lr_spec("LR0008", "wavelenghBand2", Optional[float], default=None)
    bandwidthBand2: lr_spec("LR0008", "bandwidthBand2", Optional[float], default=None)
    wavelenghBand3: lr_spec("LR0008", "wavelenghBand3", Optional[float], default=None)
    bandwidthBand3: lr_spec("LR0008", "bandwidthBand3", Optional[float], default=None)
    maxZenithAngle: lr_spec("LR0008", "maxZenithAngle", Optional[int], default=None)
    minSpectral: lr_spec("LR0008", "minSpectral", Optional[int], default=None)
    location: lr_spec("LR0008", "location", str)
    person: lr_spec("LR0008", "person", str)
    startOfCalibPeriod1: lr_spec("LR0008", "startOfCalibPeriod1", str)
    endOfCalibPeriod1: lr_spec("LR0008", "endOfCalibPeriod1", str)
    numOfComp1: lr_spec("LR0008", "numOfComp1", Optional[int], default=None)
    meanCalibCoeff1: lr_spec("LR0008", "meanCalibCoeff1", float)
    stdErrorCalibCoeff1: lr_spec("LR0008", "stdErrorCalibCoeff1", Optional[float], default=None)
    startOfCalibPeriod2: lr_spec("LR0008", "startOfCalibPeriod2", Optional[str], default=None)
    endOfCalibPeriod2: lr_spec("LR0008", "endOfCalibPeriod2", Optional[str], default=None)
    numOfComp2: lr_spec("LR0008", "numOfComp2", Optional[int], default=None)
    meanCalibCoeff2: lr_spec("LR0008", "meanCalibCoeff2", Optional[float], default=None)
    stdErrorCalibCoeff2: lr_spec("LR0008", "stdErrorCalibCoeff2", Optional[float], default=None)
    startOfCalibPeriod3: lr_spec("LR0008", "startOfCalibPeriod3", Optional[str], default=None)
    endOfCalibPeriod3: lr_spec("LR0008", "endOfCalibPeriod3", Optional[str], default=None)
    numOfComp3: lr_spec("LR0008", "numOfComp3", Optional[int], default=None)
    meanCalibCoeff3: lr_spec("LR0008", "meanCalibCoeff3", Optional[float], default=None)
    stdErrorCalibCoeff3: lr_spec("LR0008", "stdErrorCalibCoeff3", Optional[float], default=None)
    remarksOnCalib1: lr_spec("LR0008", "remarksOnCalib1", Optional[str], default=None)
    remarksOnCalib2: lr_spec("LR0008", "remarksOnCalib2", Optional[str], default=None)



def _validate_minute_vector(v, field_name: str, lr_code: str, year_month: object):
    if v is None:
        return v
    import bsrn.archive.validation as val_module

    fn = _validation_callable(
        val_module,
        "LR0100_validateFunction" if lr_code == "LR0100" else "LR4000_validateFunction",
    )
    try:
        clean = fn(v, yearMonth=year_month)
    except Exception as e:
        raise ValueError(f"{field_name}\n {str(e)}") from e
    if isinstance(v, (np.ndarray, pd.Series, list, tuple)) and clean is not v:
        return clean
    return v

class LR0100(ArchiveRecordBase):
    """
    Minute-resolution archive block; series columns accept ``pandas.Series`` or ``numpy.ndarray``.
    分钟分辨率存档块；序列列接受 ``pandas.Series`` 或 ``numpy.ndarray``。
    """

    model_config = ConfigDict(extra="ignore", frozen=False, arbitrary_types_allowed=True)

    yearMonth: lr_spec("LR0100", "yearMonth", str)
    ghi_avg: MinuteVector = lr_spec_field("LR0100", "ghi_avg", default=None)
    ghi_std: MinuteVector = lr_spec_field("LR0100", "ghi_std", default=None)
    ghi_min: MinuteVector = lr_spec_field("LR0100", "ghi_min", default=None)
    ghi_max: MinuteVector = lr_spec_field("LR0100", "ghi_max", default=None)
    bni_avg: MinuteVector = lr_spec_field("LR0100", "bni_avg", default=None)
    bni_std: MinuteVector = lr_spec_field("LR0100", "bni_std", default=None)
    bni_min: MinuteVector = lr_spec_field("LR0100", "bni_min", default=None)
    bni_max: MinuteVector = lr_spec_field("LR0100", "bni_max", default=None)
    dhi_avg: MinuteVector = lr_spec_field("LR0100", "dhi_avg", default=None)
    dhi_std: MinuteVector = lr_spec_field("LR0100", "dhi_std", default=None)
    dhi_min: MinuteVector = lr_spec_field("LR0100", "dhi_min", default=None)
    dhi_max: MinuteVector = lr_spec_field("LR0100", "dhi_max", default=None)
    lwd_avg: MinuteVector = lr_spec_field("LR0100", "lwd_avg", default=None)
    lwd_std: MinuteVector = lr_spec_field("LR0100", "lwd_std", default=None)
    lwd_min: MinuteVector = lr_spec_field("LR0100", "lwd_min", default=None)
    lwd_max: MinuteVector = lr_spec_field("LR0100", "lwd_max", default=None)
    temperature: MinuteVector = lr_spec_field("LR0100", "temperature", default=None)
    humidity: MinuteVector = lr_spec_field("LR0100", "humidity", default=None)
    pressure: MinuteVector = lr_spec_field("LR0100", "pressure", default=None)

    @field_validator(*_LR0100_MINUTE_FIELDS, mode="after")
    @classmethod
    def _validate_minute_lr0100(cls, v, info: ValidationInfo):
        return _validate_minute_vector(v, info.field_name, "LR0100", info.data.get("yearMonth"))


class LR4000(ArchiveRecordBase):
    """
    LR4000 pyrgeometer minute block; series columns accept ``pandas.Series`` or ``numpy.ndarray``.
    LR4000 长波表分钟块；序列列接受 ``pandas.Series`` 或 ``numpy.ndarray``。
    """

    model_config = ConfigDict(extra="ignore", frozen=False, arbitrary_types_allowed=True)

    yearMonth: lr_spec("LR4000", "yearMonth", str)
    domeT1_down: MinuteVector = lr_spec_field("LR4000", "domeT1_down", default=None)
    domeT2_down: MinuteVector = lr_spec_field("LR4000", "domeT2_down", default=None)
    domeT3_down: MinuteVector = lr_spec_field("LR4000", "domeT3_down", default=None)
    bodyT_down: MinuteVector = lr_spec_field("LR4000", "bodyT_down", default=None)
    longwave_down: MinuteVector = lr_spec_field("LR4000", "longwave_down", default=None)
    domeT1_up: MinuteVector = lr_spec_field("LR4000", "domeT1_up", default=None)
    domeT2_up: MinuteVector = lr_spec_field("LR4000", "domeT2_up", default=None)
    domeT3_up: MinuteVector = lr_spec_field("LR4000", "domeT3_up", default=None)
    bodyT_up: MinuteVector = lr_spec_field("LR4000", "bodyT_up", default=None)
    longwave_up: MinuteVector = lr_spec_field("LR4000", "longwave_up", default=None)

    @field_validator(*_LR4000_MINUTE_FIELDS, mode="after")
    @classmethod
    def _validate_minute_lr4000(cls, v, info: ValidationInfo):
        return _validate_minute_vector(v, info.field_name, "LR4000", info.data.get("yearMonth"))


class LR4000CONST(ArchiveRecordBase):
    serialNumber_Manufacturer: lr_spec("LR4000CONST", "serialNumber_Manufacturer", str)
    serialNumber_WRMC: lr_spec("LR4000CONST", "serialNumber_WRMC", Optional[str], default=None)
    certificateCodeID: lr_spec("LR4000CONST", "certificateCodeID", Optional[str], default=None)
    yyyymmdd: lr_spec("LR4000CONST", "yyyymmdd", Optional[int], default=None)
    manufact: lr_spec("LR4000CONST", "manufact", Optional[str], default=None)
    model: lr_spec("LR4000CONST", "model", Optional[str], default=None)
    C: lr_spec("LR4000CONST", "C", Optional[ConstScalar], default=None)
    k0: lr_spec("LR4000CONST", "k0", Optional[ConstScalar], default=None)
    k1: lr_spec("LR4000CONST", "k1", Optional[ConstScalar], default=None)
    k2: lr_spec("LR4000CONST", "k2", Optional[ConstScalar], default=None)
    k3: lr_spec("LR4000CONST", "k3", Optional[ConstScalar], default=None)
    f: lr_spec("LR4000CONST", "f", Optional[ConstScalar], default=None)


LR0001.get_bsrn_format = _FORMATTERS["LR0001"]
LR0002.get_bsrn_format = _FORMATTERS["LR0002"]
LR0003.get_bsrn_format = _FORMATTERS["LR0003"]
LR0004.get_bsrn_format = _FORMATTERS["LR0004"]
LR0005.get_bsrn_format = _FORMATTERS["LR0005"]
LR0006.get_bsrn_format = _FORMATTERS["LR0006"]
LR0007.get_bsrn_format = _FORMATTERS["LR0007"]
LR0008.get_bsrn_format = _FORMATTERS["LR0008"]
LR0100.get_bsrn_format = _FORMATTERS["LR0100"]
LR4000.get_bsrn_format = _FORMATTERS["LR4000"]
LR4000CONST.get_bsrn_format = _FORMATTERS["LR4000CONST"]

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
