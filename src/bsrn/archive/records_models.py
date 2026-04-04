"""
Explicit Pydantic logical-record models aligned with ``specs.LR_SPECS``.

Field archive metadata is attached via ``json_schema_extra`` so
:class:`~bsrn.archive.records_base.ArchiveRecordBase` validation and formatting behave
like the former ``create_model`` types.
显式 Pydantic 逻辑记录模型，与 ``specs.LR_SPECS`` 一致；通过 ``json_schema_extra``
挂接元数据，校验与格式化行为与原先动态模型相同。
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field

from .archive_lr_formats import _FORMATTERS
from .records_base import ArchiveRecordBase
from .specs import LR_SPECS

ConstScalar = Union[str, int, float]
MinuteVector = Optional[Union[pd.Series, np.ndarray]]


def af(lr: str, fname: str, **kwargs):
    """Pydantic ``Field`` tied to ``LR_SPECS[lr][fname]`` for ``_field_meta``."""
    return Field(json_schema_extra={"archive": dict(LR_SPECS[lr][fname])}, **kwargs)


class LR0001(ArchiveRecordBase):
    stationNumber: int = af("LR0001", "stationNumber")
    month: int = af("LR0001", "month")
    year: int = af("LR0001", "year")
    version: int = af("LR0001", "version")


class LR0002(ArchiveRecordBase):
    scientistChange: bool = af("LR0002", "scientistChange", default=False)
    scientistChangeDay: Optional[int] = af("LR0002", "scientistChangeDay", default=None)
    scientistChangeHour: Optional[int] = af("LR0002", "scientistChangeHour", default=None)
    scientistChangeMinute: Optional[int] = af("LR0002", "scientistChangeMinute", default=None)
    scientistName: str = af("LR0002", "scientistName")
    scientistTel: str = af("LR0002", "scientistTel")
    scientistFax: str = af("LR0002", "scientistFax")
    scientistTcpip: Optional[str] = af("LR0002", "scientistTcpip", default=None)
    scientistMail: Optional[str] = af("LR0002", "scientistMail", default=None)
    scientistAddress: str = af("LR0002", "scientistAddress")
    deputyChange: bool = af("LR0002", "deputyChange", default=False)
    deputyChangeDay: Optional[int] = af("LR0002", "deputyChangeDay", default=None)
    deputyChangeHour: Optional[int] = af("LR0002", "deputyChangeHour", default=None)
    deputyChangeMinute: Optional[int] = af("LR0002", "deputyChangeMinute", default=None)
    deputyName: str = af("LR0002", "deputyName")
    deputyTel: str = af("LR0002", "deputyTel")
    deputyFax: str = af("LR0002", "deputyFax")
    deputyTcpip: Optional[str] = af("LR0002", "deputyTcpip", default=None)
    deputyMail: Optional[str] = af("LR0002", "deputyMail", default=None)
    deputyAddress: str = af("LR0002", "deputyAddress")


class LR0003(ArchiveRecordBase):
    message: Optional[str] = af("LR0003", "message", default=None)


class LR0004(ArchiveRecordBase):
    stationDescChange: bool = af("LR0004", "stationDescChange", default=False)
    stationDescChangeDay: Optional[int] = af("LR0004", "stationDescChangeDay", default=None)
    stationDescChangeHour: Optional[int] = af("LR0004", "stationDescChangeHour", default=None)
    stationDescChangeMinute: Optional[int] = af("LR0004", "stationDescChangeMinute", default=None)
    surfaceType: int = af("LR0004", "surfaceType")
    topographyType: int = af("LR0004", "topographyType")
    address: str = af("LR0004", "address")
    telephone: Optional[str] = af("LR0004", "telephone", default=None)
    fax: Optional[str] = af("LR0004", "fax", default=None)
    tcpip: Optional[str] = af("LR0004", "tcpip", default=None)
    mail: Optional[str] = af("LR0004", "mail", default=None)
    latitude: float = af("LR0004", "latitude")
    longitude: float = af("LR0004", "longitude")
    altitude: int = af("LR0004", "altitude")
    synop: Optional[str] = af("LR0004", "synop", default=None)
    horizonChange: bool = af("LR0004", "horizonChange", default=False)
    horizonChangeDay: Optional[int] = af("LR0004", "horizonChangeDay", default=None)
    horizonChangeHour: Optional[int] = af("LR0004", "horizonChangeHour", default=None)
    horizonChangeMinute: Optional[int] = af("LR0004", "horizonChangeMinute", default=None)
    azimuth: Optional[str] = af("LR0004", "azimuth", default=None)
    elevation: Optional[str] = af("LR0004", "elevation", default=None)


class LR0005(ArchiveRecordBase):
    change: bool = af("LR0005", "change", default=False)
    changeDay: Optional[int] = af("LR0005", "changeDay", default=None)
    changeHour: Optional[int] = af("LR0005", "changeHour", default=None)
    changeMinute: Optional[int] = af("LR0005", "changeMinute", default=None)
    operating: bool = af("LR0005", "operating", default=False)
    manufacturer: str = af("LR0005", "manufacturer")
    location: str = af("LR0005", "location")
    distanceFromSite: int = af("LR0005", "distanceFromSite")
    time1stLaunch: Optional[int] = af("LR0005", "time1stLaunch", default=None)
    time2ndLaunch: Optional[int] = af("LR0005", "time2ndLaunch", default=None)
    time3rdLaunch: Optional[int] = af("LR0005", "time3rdLaunch", default=None)
    time4thLaunch: Optional[int] = af("LR0005", "time4thLaunch", default=None)
    identification: str = af("LR0005", "identification")
    remarks: Optional[str] = af("LR0005", "remarks", default=None)


class LR0006(ArchiveRecordBase):
    change: bool = af("LR0006", "change", default=False)
    changeDay: Optional[int] = af("LR0006", "changeDay", default=None)
    changeHour: Optional[int] = af("LR0006", "changeHour", default=None)
    changeMinute: Optional[int] = af("LR0006", "changeMinute", default=None)
    operating: bool = af("LR0006", "operating", default=False)
    manufacturer: str = af("LR0006", "manufacturer")
    location: str = af("LR0006", "location")
    distanceFromSite: int = af("LR0006", "distanceFromSite")
    identification: str = af("LR0006", "identification")
    remarks: Optional[str] = af("LR0006", "remarks", default=None)


class LR0007(ArchiveRecordBase):
    change: bool = af("LR0007", "change", default=False)
    changeDay: Optional[int] = af("LR0007", "changeDay", default=None)
    changeHour: Optional[int] = af("LR0007", "changeHour", default=None)
    changeMinute: Optional[int] = af("LR0007", "changeMinute", default=None)
    cloudAmount: Optional[str] = af("LR0007", "cloudAmount", default=None)
    cloudBaseHeight: Optional[str] = af("LR0007", "cloudBaseHeight", default=None)
    cloudLiquid: Optional[str] = af("LR0007", "cloudLiquid", default=None)
    cloudAerosol: Optional[str] = af("LR0007", "cloudAerosol", default=None)
    waterVapour: Optional[str] = af("LR0007", "waterVapour", default=None)


class LR0008(ArchiveRecordBase):
    change: bool = af("LR0008", "change", default=False)
    changeDay: Optional[int] = af("LR0008", "changeDay", default=None)
    changeHour: Optional[int] = af("LR0008", "changeHour", default=None)
    changeMinute: Optional[int] = af("LR0008", "changeMinute", default=None)
    operating: bool = af("LR0008", "operating", default=False)
    radiationQuantityMeasured: int = af("LR0008", "radiationQuantityMeasured")
    manufacturer: str = af("LR0008", "manufacturer")
    model: str = af("LR0008", "model")
    serialNumber: str = af("LR0008", "serialNumber")
    dateOfPurchase: Optional[str] = af("LR0008", "dateOfPurchase", default=None)
    identification: int = af("LR0008", "identification")
    remarks: Optional[str] = af("LR0008", "remarks", default=None)
    pyrgeometerBody: Optional[int] = af("LR0008", "pyrgeometerBody", default=None)
    pyrgeometerDome: Optional[int] = af("LR0008", "pyrgeometerDome", default=None)
    numOfBand: Optional[int] = af("LR0008", "numOfBand", default=None)
    wavelenghBand1: Optional[float] = af("LR0008", "wavelenghBand1", default=None)
    bandwidthBand1: Optional[float] = af("LR0008", "bandwidthBand1", default=None)
    wavelenghBand2: Optional[float] = af("LR0008", "wavelenghBand2", default=None)
    bandwidthBand2: Optional[float] = af("LR0008", "bandwidthBand2", default=None)
    wavelenghBand3: Optional[float] = af("LR0008", "wavelenghBand3", default=None)
    bandwidthBand3: Optional[float] = af("LR0008", "bandwidthBand3", default=None)
    maxZenithAngle: Optional[int] = af("LR0008", "maxZenithAngle", default=None)
    minSpectral: Optional[int] = af("LR0008", "minSpectral", default=None)
    location: str = af("LR0008", "location")
    person: str = af("LR0008", "person")
    startOfCalibPeriod1: str = af("LR0008", "startOfCalibPeriod1")
    endOfCalibPeriod1: str = af("LR0008", "endOfCalibPeriod1")
    numOfComp1: Optional[int] = af("LR0008", "numOfComp1", default=None)
    meanCalibCoeff1: float = af("LR0008", "meanCalibCoeff1")
    stdErrorCalibCoeff1: Optional[float] = af("LR0008", "stdErrorCalibCoeff1", default=None)
    startOfCalibPeriod2: Optional[str] = af("LR0008", "startOfCalibPeriod2", default=None)
    endOfCalibPeriod2: Optional[str] = af("LR0008", "endOfCalibPeriod2", default=None)
    numOfComp2: Optional[int] = af("LR0008", "numOfComp2", default=None)
    meanCalibCoeff2: Optional[float] = af("LR0008", "meanCalibCoeff2", default=None)
    stdErrorCalibCoeff2: Optional[float] = af("LR0008", "stdErrorCalibCoeff2", default=None)
    startOfCalibPeriod3: Optional[str] = af("LR0008", "startOfCalibPeriod3", default=None)
    endOfCalibPeriod3: Optional[str] = af("LR0008", "endOfCalibPeriod3", default=None)
    numOfComp3: Optional[int] = af("LR0008", "numOfComp3", default=None)
    meanCalibCoeff3: Optional[float] = af("LR0008", "meanCalibCoeff3", default=None)
    stdErrorCalibCoeff3: Optional[float] = af("LR0008", "stdErrorCalibCoeff3", default=None)
    remarksOnCalib1: Optional[str] = af("LR0008", "remarksOnCalib1", default=None)
    remarksOnCalib2: Optional[str] = af("LR0008", "remarksOnCalib2", default=None)


class LR0100(ArchiveRecordBase):
    """
    Minute-resolution archive block; series columns accept ``pandas.Series`` or ``numpy.ndarray``.
    分钟分辨率存档块；序列列接受 ``pandas.Series`` 或 ``numpy.ndarray``。
    """

    model_config = ConfigDict(extra="ignore", frozen=False, arbitrary_types_allowed=True)

    yearMonth: str = af("LR0100", "yearMonth")
    ghi_avg: MinuteVector = af("LR0100", "ghi_avg", default=None)
    ghi_std: MinuteVector = af("LR0100", "ghi_std", default=None)
    ghi_min: MinuteVector = af("LR0100", "ghi_min", default=None)
    ghi_max: MinuteVector = af("LR0100", "ghi_max", default=None)
    bni_avg: MinuteVector = af("LR0100", "bni_avg", default=None)
    bni_std: MinuteVector = af("LR0100", "bni_std", default=None)
    bni_min: MinuteVector = af("LR0100", "bni_min", default=None)
    bni_max: MinuteVector = af("LR0100", "bni_max", default=None)
    dhi_avg: MinuteVector = af("LR0100", "dhi_avg", default=None)
    dhi_std: MinuteVector = af("LR0100", "dhi_std", default=None)
    dhi_min: MinuteVector = af("LR0100", "dhi_min", default=None)
    dhi_max: MinuteVector = af("LR0100", "dhi_max", default=None)
    lwd_avg: MinuteVector = af("LR0100", "lwd_avg", default=None)
    lwd_std: MinuteVector = af("LR0100", "lwd_std", default=None)
    lwd_min: MinuteVector = af("LR0100", "lwd_min", default=None)
    lwd_max: MinuteVector = af("LR0100", "lwd_max", default=None)
    temperature: MinuteVector = af("LR0100", "temperature", default=None)
    humidity: MinuteVector = af("LR0100", "humidity", default=None)
    pressure: MinuteVector = af("LR0100", "pressure", default=None)


class LR4000(ArchiveRecordBase):
    """
    LR4000 pyrgeometer minute block; series columns accept ``pandas.Series`` or ``numpy.ndarray``.
    LR4000 长波表分钟块；序列列接受 ``pandas.Series`` 或 ``numpy.ndarray``。
    """

    model_config = ConfigDict(extra="ignore", frozen=False, arbitrary_types_allowed=True)

    yearMonth: str = af("LR4000", "yearMonth")
    domeT1_down: MinuteVector = af("LR4000", "domeT1_down", default=None)
    domeT2_down: MinuteVector = af("LR4000", "domeT2_down", default=None)
    domeT3_down: MinuteVector = af("LR4000", "domeT3_down", default=None)
    bodyT_down: MinuteVector = af("LR4000", "bodyT_down", default=None)
    longwave_down: MinuteVector = af("LR4000", "longwave_down", default=None)
    domeT1_up: MinuteVector = af("LR4000", "domeT1_up", default=None)
    domeT2_up: MinuteVector = af("LR4000", "domeT2_up", default=None)
    domeT3_up: MinuteVector = af("LR4000", "domeT3_up", default=None)
    bodyT_up: MinuteVector = af("LR4000", "bodyT_up", default=None)
    longwave_up: MinuteVector = af("LR4000", "longwave_up", default=None)


class LR4000CONST(ArchiveRecordBase):
    serialNumber_Manufacturer: str = af("LR4000CONST", "serialNumber_Manufacturer")
    serialNumber_WRMC: Optional[str] = af("LR4000CONST", "serialNumber_WRMC", default=None)
    certificateCodeID: Optional[str] = af("LR4000CONST", "certificateCodeID", default=None)
    yyyymmdd: Optional[int] = af("LR4000CONST", "yyyymmdd", default=None)
    manufact: Optional[str] = af("LR4000CONST", "manufact", default=None)
    model: Optional[str] = af("LR4000CONST", "model", default=None)
    C: Optional[ConstScalar] = af("LR4000CONST", "C", default=None)
    k0: Optional[ConstScalar] = af("LR4000CONST", "k0", default=None)
    k1: Optional[ConstScalar] = af("LR4000CONST", "k1", default=None)
    k2: Optional[ConstScalar] = af("LR4000CONST", "k2", default=None)
    k3: Optional[ConstScalar] = af("LR4000CONST", "k3", default=None)
    f: Optional[ConstScalar] = af("LR4000CONST", "f", default=None)


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
