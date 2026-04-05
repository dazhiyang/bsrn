"""
``get_bsrn_format`` implementations for Pydantic archive LR models (R ``2_R6Class_*.R``).
Pydantic 存档逻辑记录的 ``get_bsrn_format`` 实现（R ``2_R6Class_*.R``）。
"""

import calendar
import textwrap

import numpy as np
import pandas as pd

from .api import get_azimuth_elevation


def lr0001_get_bsrn_format(self, listSensor=None):
    """Emit ``*C0001`` block. / 输出 ``*C0001`` 块。"""
    if listSensor is None:
        listSensor = ["2", "3", "4", "5", "21", "22", "23"]
    self.stop_if_values_missing("LR0001")
    ls = [int(x) for x in listSensor]
    n = len(ls)
    pad = 8 - (n % 8) if (n % 8) != 0 else 0
    listIds = ls + [-1] * pad
    rows = [" ".join([f"{x:>9}" for x in listIds[i : i + 8]]) for i in range(0, len(listIds), 8)]
    formatListSensor = " " + "\n ".join(rows)
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    return (
        f"*C0001\n"
        f" {v['stationNumber']} {v['month']} {v['year']} {v['version']}\n"
        f"{formatListSensor}"
    )


def lr0002_get_bsrn_format(self):
    """Emit LR0002 contact block. / 输出 LR0002 联系信息块。"""
    self.stop_if_values_missing("LR0002")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    sci_change = self._private["scientistChange"]
    dep_change = self._private["deputyChange"]
    c1 = "*C0002" if (sci_change or dep_change) else "*U0002"
    c2 = (
        f" {v['scientistChangeDay']} {v['scientistChangeHour']} {v['scientistChangeMinute']}"
        if sci_change
        else " -1 -1 -1"
    )
    c3 = f"{v['scientistName']} {v['scientistTel']} {v['scientistFax']}"
    c4 = f"{v['scientistTcpip']} {v['scientistMail']}"
    c5 = f"{v['scientistAddress']}"
    c6 = (
        f" {v['deputyChangeDay']} {v['deputyChangeHour']} {v['deputyChangeMinute']}"
        if dep_change
        else " -1 -1 -1"
    )
    c7 = f"{v['deputyName']} {v['deputyTel']} {v['deputyFax']}"
    c8 = f"{v['deputyTcpip']} {v['deputyMail']}"
    c9 = f"{v['deputyAddress']}"
    return "\n".join([c1, c2, c3, c4, c5, c6, c7, c8, c9])


def lr0003_get_bsrn_format(self, *args):
    """Emit LR0003 message block. / 输出 LR0003 消息块。"""
    self.stop_if_values_missing("LR0003")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    res = "*U0003\n" + v["message"]
    if args:
        res += "\n" + "\n".join(args)
    return res


def lr0004_get_bsrn_format(self):
    """Emit LR0004 block. / 输出 LR0004 块。"""
    self.stop_if_values_missing("LR0004")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    s_change = self._private["stationDescChange"]
    h_change = self._private["horizonChange"]
    c1 = "*C0004" if (s_change or h_change) else "*U0004"
    c2 = (
        f" {v['stationDescChangeDay']} {v['stationDescChangeHour']} {v['stationDescChangeMinute']}"
        if s_change
        else " -1 -1 -1"
    )
    c3 = f" {v['surfaceType']} {v['topographyType']}"
    c4 = f"{v['address']}"
    c5 = f"{v['telephone']} {v['fax']}"
    c6 = f"{v['tcpip']} {v['mail']}"
    c7 = f" {v['latitude']} {v['longitude']} {v['altitude']} {v['synop']}"
    c8 = (
        f" {v['horizonChangeDay']} {v['horizonChangeHour']} {v['horizonChangeMinute']}"
        if h_change
        else " -1 -1 -1"
    )
    c9 = get_azimuth_elevation(self._private["azimuth"], self._private["elevation"])
    return "\n".join([c1, c2, c3, c4, c5, c6, c7, c8, c9])


def lr0005_get_bsrn_format(self):
    """Emit LR0005 radiosonde block. / 输出 LR0005 探空块。"""
    self.stop_if_values_missing("LR0005")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    ch = self._private["change"]
    c1 = "*C0005" if ch else "*U0005"
    c2_t = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
    c2_o = "Y" if self._private["operating"] else "N"
    c2 = f"{c2_t} {c2_o}"
    c3 = (
        f"{v['manufacturer']} {v['location']} {v['distanceFromSite']} "
        f"{v['time1stLaunch']} {v['time2ndLaunch']} {v['time3rdLaunch']} {v['time4thLaunch']} "
        f"{v['identification']}"
    )
    c4 = f"{v['remarks']}"
    return "\n".join([c1, c2, c3, c4])


def lr0006_get_bsrn_format(self):
    """Emit LR0006 ozone block. / 输出 LR0006 臭氧块。"""
    self.stop_if_values_missing("LR0006")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    ch = self._private["change"]
    c1 = "*C0006" if ch else "*U0006"
    c2_t = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
    c2_o = "Y" if self._private["operating"] else "N"
    c2 = f"{c2_t} {c2_o}"
    c3 = f"{v['manufacturer']} {v['location']} {v['distanceFromSite']} {v['identification']}"
    c4 = f"{v['remarks']}"
    return "\n".join([c1, c2, c3, c4])


def lr0007_get_bsrn_format(self, synop=None):
    """Emit LR0007 block. / 输出 LR0007 块。"""
    self.stop_if_values_missing("LR0007")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    flags = [
        "N" if synop is None else "Y",
        "N" if self._private["cloudAmount"] is None else "Y",
        "N" if self._private["cloudBaseHeight"] is None else "Y",
        "N" if self._private["cloudLiquid"] is None else "Y",
        "N" if self._private["cloudAerosol"] is None else "Y",
        "N" if self._private["waterVapour"] is None else "Y",
    ]
    flags_str = " ".join(flags)
    ch = self._private["change"]
    c1 = "*C0007" if ch else "*U0007"
    c2 = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
    c3 = f"{v['cloudAmount']}"
    c4 = f"{v['cloudBaseHeight']}"
    c5 = f"{v['cloudLiquid']}"
    c6 = f"{v['cloudAerosol']}"
    c7 = f"{v['waterVapour']}"
    return "\n".join([c1, c2, c3, c4, c5, c6, c7, flags_str])


def lr0008_get_bsrn_format(self, anyChange=False, printLr=False, LR0009Format=False):
    """Emit LR0008 or LR0009-style fragment. / 输出 LR0008 或 LR0009 式片段。"""
    self.stop_if_values_missing("LR0008")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    ch = self._private["change"]
    t_str = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
    if LR0009Format:
        rq = int(self._private["radiationQuantityMeasured"])
        ident = int(self._private["identification"])
        nb = self._private["numOfBand"]
        if nb is None:
            nb = -1
        thisFormat = f"{t_str}         {rq} {ident} {nb}"
    else:
        c1 = f"{t_str} {'Y' if self._private['operating'] else 'N'}"
        c2 = (
            f"{v['manufacturer']} {v['model']} {v['serialNumber']} {v['dateOfPurchase']} "
            f"{v['identification']}"
        )
        c3 = f"{v['remarks']}"
        c4 = (
            f" {v['pyrgeometerBody']} {v['pyrgeometerDome']} {v['wavelenghBand1']} "
            f"{v['bandwidthBand1']} {v['wavelenghBand2']} {v['bandwidthBand2']} "
            f"{v['wavelenghBand3']} {v['bandwidthBand3']} {v['maxZenithAngle']} {v['minSpectral']}"
        )
        c5 = f"{v['location']} {v['person']}"
        c6 = (
            f"{v['startOfCalibPeriod1']} {v['endOfCalibPeriod1']} {v['numOfComp1']} "
            f"{v['meanCalibCoeff1']} {v['stdErrorCalibCoeff1']}"
        )
        c7 = (
            f"{v['startOfCalibPeriod2']} {v['endOfCalibPeriod2']} {v['numOfComp2']} "
            f"{v['meanCalibCoeff2']} {v['stdErrorCalibCoeff2']}"
        )
        c8 = (
            f"{v['startOfCalibPeriod3']} {v['endOfCalibPeriod3']} {v['numOfComp3']} "
            f"{v['meanCalibCoeff3']} {v['stdErrorCalibCoeff3']}"
        )
        c9 = f"{v['remarksOnCalib1']}"
        c10 = f"{v['remarksOnCalib2']}"
        thisFormat = "\n".join([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    if printLr:
        h = f"{'*C000' if anyChange else '*U000'}{'9' if LR0009Format else '8'}\n"
        thisFormat = h + thisFormat
    return thisFormat


def lr4000const_get_bsrn_format(self, method=1):
    """Emit ``@LR4000CONST`` wrapped line. / 输出折行 ``@LR4000CONST``。"""
    self.stop_if_values_missing("LR4000CONST")
    if method not in [1, 2]:
        raise ValueError("method must be 1 or 2")
    v = {name: self.get_format_value(name) for name in self._params.keys()}
    cert_id = self._private["certificateCodeID"]
    if method == 2:
        if not self._private["yyyymmdd"] or not self._private["manufact"] or not self._private["model"]:
            raise ValueError("missing value(s) : yyyymmdd, manufact or model")
        cert_id = (
            f"CAL_{self._private['yyyymmdd']}_{self._private['manufact']}_{self._private['model']}_"
            f"{self._private['serialNumber_Manufacturer']}_{self._private['serialNumber_WRMC']}"
        )
    if not cert_id:
        raise ValueError("missing value(s) : certificateCodeID")
    s = (
        f"@LR4000CONST, {v['serialNumber_Manufacturer']}, {v['serialNumber_WRMC']}, {cert_id}, "
        f"{v['C']}, {v['k0']}, {v['k1']}, {v['k2']}, {v['k3']}, {v['f']}"
    )
    return textwrap.fill(s, width=79).replace("\n", "&\n")


def lr0100_get_bsrn_format(self, changed=True):
    """Emit full LR0100 minute table. / 输出 LR0100 分钟表。"""
    res = "*C0100" if changed else "*U0100"
    m = self._format_series_field
    y, mo = map(int, self._private["yearMonth"].split("-"))
    nd = calendar.monthrange(y, mo)[1]
    days = np.repeat(np.arange(1, nd + 1), 1440)
    df_days = pd.Series([f"{d:>2d}" for d in days])
    mins = np.tile(np.arange(0, 1440), nd)
    df_mins = pd.Series([f"{m_:>4d}" for m_ in mins])
    line1 = (
        " " + df_days + " " + df_mins + "   "
        + m("ghi_avg") + " " + m("ghi_std") + " " + m("ghi_min") + " " + m("ghi_max") + "   "
        + m("bni_avg") + " " + m("bni_std") + " " + m("bni_min") + " " + m("bni_max")
    )
    line2 = (
        "           "
        + m("dhi_avg") + " " + m("dhi_std") + " " + m("dhi_min") + " " + m("dhi_max") + "   "
        + m("lwd_avg") + " " + m("lwd_std") + " " + m("lwd_min") + " " + m("lwd_max")
        + "    "
        + m("temperature") + " " + m("humidity") + " " + m("pressure")
    )
    strData = (line1 + "\n" + line2).str.cat(sep="\n")
    return f"{res}\n{strData}"


def lr4000_get_bsrn_format(self, changed=True):
    """Emit full LR4000 minute table. / 输出 LR4000 分钟表。"""
    res = "*C4000" if changed else "*U4000"
    m = self._format_series_field
    y, mo = map(int, self._private["yearMonth"].split("-"))
    nd = calendar.monthrange(y, mo)[1]
    days = np.repeat(np.arange(1, nd + 1), 1440)
    df_days = pd.Series([f"{d:>2d}" for d in days])
    mins = np.tile(np.arange(0, 1440), nd)
    df_mins = pd.Series([f"{mi:>4d}" for mi in mins])
    strData = (
        " " + df_days + " " + df_mins + " "
        + m("domeT1_down") + " " + m("domeT2_down") + " " + m("domeT3_down") + " "
        + m("bodyT_down") + " " + m("longwave_down") + "  "
        + m("domeT1_up") + " " + m("domeT2_up") + " " + m("domeT3_up") + " "
        + m("bodyT_up") + " " + m("longwave_up")
    ).str.cat(sep="\n")
    return f"{res}\n{strData}"


_FORMATTERS = {
    "LR0001": lr0001_get_bsrn_format,
    "LR0002": lr0002_get_bsrn_format,
    "LR0003": lr0003_get_bsrn_format,
    "LR0004": lr0004_get_bsrn_format,
    "LR0005": lr0005_get_bsrn_format,
    "LR0006": lr0006_get_bsrn_format,
    "LR0007": lr0007_get_bsrn_format,
    "LR0008": lr0008_get_bsrn_format,
    "LR4000CONST": lr4000const_get_bsrn_format,
    "LR0100": lr0100_get_bsrn_format,
    "LR4000": lr4000_get_bsrn_format,
}
