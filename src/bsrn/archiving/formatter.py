"""
BSRN logical-record classes (``LR0001`` … ``LR4000``) and archive ASCII output.

BSRN 逻辑记录类 (``LR0001`` … ``LR4000``) 和 ASCII 输出格式化。
"""

import calendar
import textwrap

import numpy as np
import pandas as pd

from .api import BSRNRecord, get_azimuth_elevation

# =============================================================================
# TRANSLATION OF: 2_R6Class_headers.R
# =============================================================================


class LR0001(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0001", **kwargs)

    def get_bsrn_format(self, listSensor=["2", "3", "4", "5", "21", "22", "23"]):
        """Translates lr0001GetBsrnFormat"""
        self.stop_if_values_missing("LR0001")

        ls = [int(x) for x in listSensor]
        n = len(ls)
        pad = 8 - (n % 8) if (n % 8) != 0 else 0
        listIds = ls + [-1] * pad

        rows = [" ".join([f"{x:>9}" for x in listIds[i:i+8]]) for i in range(0, len(listIds), 8)]
        formatListSensor = " " + "\n ".join(rows)

        v = {name: self.get_format_value(name) for name in self._params.keys()}

        return (
            f"*C0001\n"
            f" {v['stationNumber']} {v['month']} {v['year']} {v['version']}\n"
            f"{formatListSensor}"
        )


class LR0002(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0002", **kwargs)

    def get_bsrn_format(self):
        """Translates lr0002GetBsrnFormat"""
        self.stop_if_values_missing("LR0002")
        v = {name: self.get_format_value(name) for name in self._params.keys()}

        sci_change = self._private['scientistChange']
        dep_change = self._private['deputyChange']

        c1 = "*C0002" if (sci_change or dep_change) else "*U0002"
        c2 = f" {v['scientistChangeDay']} {v['scientistChangeHour']} {v['scientistChangeMinute']}" if sci_change else " -1 -1 -1"
        c3 = f"{v['scientistName']} {v['scientistTel']} {v['scientistFax']}"
        c4 = f"{v['scientistTcpip']} {v['scientistMail']}"
        c5 = f"{v['scientistAddress']}"
        c6 = f" {v['deputyChangeDay']} {v['deputyChangeHour']} {v['deputyChangeMinute']}" if dep_change else " -1 -1 -1"
        c7 = f"{v['deputyName']} {v['deputyTel']} {v['deputyFax']}"
        c8 = f"{v['deputyTcpip']} {v['deputyMail']}"
        c9 = f"{v['deputyAddress']}"

        return "\n".join([c1, c2, c3, c4, c5, c6, c7, c8, c9])


class LR0003(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0003", **kwargs)

    def get_bsrn_format(self, *args):
        """Translates lr0003GetBsrnFormat"""
        self.stop_if_values_missing("LR0003")
        v = {name: self.get_format_value(name) for name in self._params.keys()}
        res = "*U0003\n" + v['message']
        if args:
            res += "\n" + "\n".join(args)
        return res


class LR0004(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0004", **kwargs)

    def get_bsrn_format(self):
        """Translates lr0004GetBsrnFormat"""
        self.stop_if_values_missing("LR0004")
        v = {name: self.get_format_value(name) for name in self._params.keys()}

        s_change = self._private['stationDescChange']
        h_change = self._private['horizonChange']

        c1 = "*C0004" if (s_change or h_change) else "*U0004"
        c2 = f" {v['stationDescChangeDay']} {v['stationDescChangeHour']} {v['stationDescChangeMinute']}" if s_change else " -1 -1 -1"
        c3 = f" {v['surfaceType']} {v['topographyType']}"
        c4 = f"{v['address']}"
        c5 = f"{v['telephone']} {v['fax']}"
        c6 = f"{v['tcpip']} {v['mail']}"
        c7 = f" {v['latitude']} {v['longitude']} {v['altitude']} {v['synop']}"
        c8 = f" {v['horizonChangeDay']} {v['horizonChangeHour']} {v['horizonChangeMinute']}" if h_change else " -1 -1 -1"
        c9 = get_azimuth_elevation(self._private['azimuth'], self._private['elevation'])

        return "\n".join([c1, c2, c3, c4, c5, c6, c7, c8, c9])


class LR0005(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0005", **kwargs)

    def get_bsrn_format(self):
        """Translates lr0005GetBsrnFormat"""
        self.stop_if_values_missing("LR0005")
        v = {name: self.get_format_value(name) for name in self._params.keys()}

        ch = self._private['change']
        c1 = "*C0005" if ch else "*U0005"
        c2_t = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
        c2_o = "Y" if self._private['operating'] else "N"
        c2 = f"{c2_t} {c2_o}"
        c3 = f"{v['manufacturer']} {v['location']} {v['distanceFromSite']} {v['time1stLaunch']} {v['time2ndLaunch']} {v['time3rdLaunch']} {v['time4thLaunch']} {v['identification']}"
        c4 = f"{v['remarks']}"

        return "\n".join([c1, c2, c3, c4])


class LR0006(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0006", **kwargs)

    def get_bsrn_format(self):
        """Translates lr0006GetBsrnFormat"""
        self.stop_if_values_missing("LR0006")
        v = {name: self.get_format_value(name) for name in self._params.keys()}

        ch = self._private['change']
        c1 = "*C0006" if ch else "*U0006"
        c2_t = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
        c2_o = "Y" if self._private['operating'] else "N"
        c2 = f"{c2_t} {c2_o}"
        c3 = f"{v['manufacturer']} {v['location']} {v['distanceFromSite']} {v['identification']}"
        c4 = f"{v['remarks']}"

        return "\n".join([c1, c2, c3, c4])


class LR0007(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0007", **kwargs)

    def get_bsrn_format(self, synop=None):
        """Translates lr0007GetBsrnFormat"""
        self.stop_if_values_missing("LR0007")
        v = {name: self.get_format_value(name) for name in self._params.keys()}

        flags = [
            "N" if synop is None else "Y",
            "N" if self._private['cloudAmount'] is None else "Y",
            "N" if self._private['cloudBaseHeight'] is None else "Y",
            "N" if self._private['cloudLiquid'] is None else "Y",
            "N" if self._private['cloudAerosol'] is None else "Y",
            "N" if self._private['waterVapour'] is None else "Y"
        ]
        flags_str = " ".join(flags)

        ch = self._private['change']
        c1 = "*C0007" if ch else "*U0007"
        c2 = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"
        c3 = f"{v['cloudAmount']}"
        c4 = f"{v['cloudBaseHeight']}"
        c5 = f"{v['cloudLiquid']}"
        c6 = f"{v['cloudAerosol']}"
        c7 = f"{v['waterVapour']}"

        return "\n".join([c1, c2, c3, c4, c5, c6, c7, flags_str])


class LR0008(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0008", **kwargs)

    def get_bsrn_format(self, anyChange=False, printLr=False, LR0009Format=False):
        """Translates lr0008GetBsrnFormat"""
        self.stop_if_values_missing("LR0008")
        v = {name: self.get_format_value(name) for name in self._params.keys()}

        ch = self._private['change']
        t_str = f" {v['changeDay']} {v['changeHour']} {v['changeMinute']}" if ch else " -1 -1 -1"

        if LR0009Format:
            # One physical line: ``t_str`` + 9 spaces + quantity + id + band (matches R / WRMC reference files).
            rq = int(self._private["radiationQuantityMeasured"])
            ident = int(self._private["identification"])
            nb = self._private["numOfBand"]
            if nb is None:
                nb = -1
            thisFormat = f"{t_str}         {rq} {ident} {nb}"
        else:
            c1 = f"{t_str} {'Y' if self._private['operating'] else 'N'}"
            c2 = f"{v['manufacturer']} {v['model']} {v['serialNumber']} {v['dateOfPurchase']} {v['identification']}"
            c3 = f"{v['remarks']}"
            c4 = f" {v['pyrgeometerBody']} {v['pyrgeometerDome']} {v['wavelenghBand1']} {v['bandwidthBand1']} {v['wavelenghBand2']} {v['bandwidthBand2']} {v['wavelenghBand3']} {v['bandwidthBand3']} {v['maxZenithAngle']} {v['minSpectral']}"
            c5 = f"{v['location']} {v['person']}"
            c6 = f"{v['startOfCalibPeriod1']} {v['endOfCalibPeriod1']} {v['numOfComp1']} {v['meanCalibCoeff1']} {v['stdErrorCalibCoeff1']}"
            c7 = f"{v['startOfCalibPeriod2']} {v['endOfCalibPeriod2']} {v['numOfComp2']} {v['meanCalibCoeff2']} {v['stdErrorCalibCoeff2']}"
            c8 = f"{v['startOfCalibPeriod3']} {v['endOfCalibPeriod3']} {v['numOfComp3']} {v['meanCalibCoeff3']} {v['stdErrorCalibCoeff3']}"
            c9 = f"{v['remarksOnCalib1']}"
            c10 = f"{v['remarksOnCalib2']}"
            thisFormat = "\n".join([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])

        if printLr:
            h = f"{'*C000' if anyChange else '*U000'}{'9' if LR0009Format else '8'}\n"
            thisFormat = h + thisFormat

        return thisFormat


class LR4000CONST(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR4000CONST", **kwargs)

    def get_bsrn_format(self, method=1):
        """Translates lr4000constGetBsrnFormat"""
        self.stop_if_values_missing("LR4000CONST")
        if method not in [1, 2]:
            raise ValueError("method must be 1 or 2")

        v = {name: self.get_format_value(name) for name in self._params.keys()}

        cert_id = self._private['certificateCodeID']
        if method == 2:
            if not self._private['yyyymmdd'] or not self._private['manufact'] or not self._private['model']:
                raise ValueError("missing value(s) : yyyymmdd, manufact or model")
            cert_id = f"CAL_{self._private['yyyymmdd']}_{self._private['manufact']}_{self._private['model']}_{self._private['serialNumber_Manufacturer']}_{self._private['serialNumber_WRMC']}"

        if not cert_id:
            raise ValueError("missing value(s) : certificateCodeID")

        s = f"@LR4000CONST, {v['serialNumber_Manufacturer']}, {v['serialNumber_WRMC']}, {cert_id}, {v['C']}, {v['k0']}, {v['k1']}, {v['k2']}, {v['k3']}, {v['f']}"
        res = textwrap.fill(s, width=79).replace('\n', '&\n')
        return res


# =============================================================================
# TRANSLATION OF: 2_R6Class_datas.R
# =============================================================================


class LR0100(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR0100", **kwargs)

    def get_bsrn_format(self, changed=True):
        """Translates lr0100GetBsrnFormat"""
        res = "*C0100" if changed else "*U0100"
        m = self._format_series_field

        y, mo = map(int, self._private["yearMonth"].split("-"))
        nd = calendar.monthrange(y, mo)[1]

        # Replicates: rep(1:nd, each = 1440) %>% formatC(format = 'd', width = 2)
        days = np.repeat(np.arange(1, nd + 1), 1440)
        df_days = pd.Series([f"{d:>2d}" for d in days])

        # Replicates: rep(0:1439, nd) %>%  formatC(format = 'd', width = 4)
        mins = np.tile(np.arange(0, 1440), nd)
        df_mins = pd.Series([f"{m_:>4d}" for m_ in mins])

        line1 = (
            " " + df_days + " " + df_mins + "   "
            + m("global2_avg") + " " + m("global2_std") + " " + m("global2_min") + " " + m("global2_max") + "   "
            + m("direct_avg") + " " + m("direct_std") + " " + m("direct_min") + " " + m("direct_max")
        )

        line2 = (
            "           "
            + m("diffuse_avg") + " " + m("diffuse_std") + " " + m("diffuse_min") + " " + m("diffuse_max") + "   "
            + m("downward_avg") + " " + m("downward_std") + " " + m("downward_min") + " " + m("downward_max")
            + "    "
            + m("temperature") + " " + m("humidity") + " " + m("pressure")
        )

        strData = (line1 + "\n" + line2).str.cat(sep="\n")
        return f"{res}\n{strData}"


class LR4000(BSRNRecord):
    def __init__(self, **kwargs):
        super().__init__("LR4000", **kwargs)

    def get_bsrn_format(self, changed=True):
        """Translates lr4000GetBsrnFormat"""
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


# =============================================================================
# ``0_bsrnFormats_headers.R`` — LR0001–LR0008, LR4000CONST
# R ``0_bsrnFormats_headers.R``：头记录 LR0001–LR0008 与 LR4000CONST
# =============================================================================


def lr0001_format(v, listSensor=None):
    """
    Format LR0001 header lines (R ``lr0001GetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0001`` (see ``specs.LR_SPECS``).
        ``LR0001`` 字段字典（见 ``specs.LR_SPECS``）。
    listSensor : list of str, optional
        Sensor slot codes included in the formatted line.
        参与格式化的传感器槽位代码列表。

    Returns
    -------
    str
        BSRN-formatted LR0001 text block.
        BSRN 格式的 LR0001 文本块。
    """
    if listSensor is None:
        listSensor = ["2", "3", "4", "5", "21", "22", "23"]
    return LR0001(**v).get_bsrn_format(listSensor=listSensor)


def lr0002_format(v, _scientistChange, _deputyChange):
    """
    Format LR0002 header lines (R ``lr0002GetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0002``; scientist/deputy change flags live here.
        ``LR0002`` 字段字典；科学家/副手变更标志在 ``v`` 内。
    _scientistChange, _deputyChange : Any
        Unused placeholders for older call signatures; ignored.
        兼容旧调用签名的占位参数，不使用。

    Returns
    -------
    str
        BSRN-formatted LR0002 text block.
        BSRN 格式的 LR0002 文本块。
    """
    return LR0002(**v).get_bsrn_format()


def lr0003_format(v, *extra_blocks):
    """
    Format LR0003 free-text / comment block (R ``lr0003GetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0003`` (typically ``message``).
        ``LR0003`` 字段（一般为 ``message``）。
    *extra_blocks : str
        Optional extra lines appended after the message (R ``...`` args).
        可选的附加行，接在消息正文之后（R 中 ``...`` 参数）。

    Returns
    -------
    str
        BSRN-formatted LR0003 text block.
        BSRN 格式的 LR0003 文本块。
    """
    return LR0003(**v).get_bsrn_format(*extra_blocks)


def lr0004_format(v, _stationDescChange, _horizonChange, azimuth, elevation):
    """
    Format LR0004 station-description block (R ``lr0004GetBsrnFormat``).

    Uses ``stationDescChangeMinute`` / ``horizonChangeMinute`` keys in ``v`` (R ``...Minute``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0004`` except horizon/azimuth vectors.
        ``LR0004`` 字段，不含方位/高度向量。
    _stationDescChange, _horizonChange : Any
        Unused placeholders for older call signatures; ignored.
        兼容旧调用签名的占位参数，不使用。
    azimuth : str or sequence
        Azimuth list string ``A1,A2,...`` or equivalent.
        方位角列表字符串 ``A1,A2,...`` 或同类输入。
    elevation : str or sequence
        Elevation list string ``E1,E2,...`` or equivalent.
        高度角列表字符串 ``E1,E2,...`` 或同类输入。

    Returns
    -------
    str
        BSRN-formatted LR0004 text block.
        BSRN 格式的 LR0004 文本块。
    """
    # Merge horizon fields with station metadata. / 合并台站元数据与地平线向量。
    merged = {**v, "azimuth": azimuth, "elevation": elevation}
    return LR0004(**merged).get_bsrn_format()


def lr0005_format(v):
    """
    Format LR0005 radiosonde metadata (R ``lr0005GetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0005`` (see ``specs.LR_SPECS``).
        ``LR0005`` 字段字典（见 ``specs.LR_SPECS``）。

    Returns
    -------
    str
        BSRN-formatted LR0005 text block.
        BSRN 格式的 LR0005 文本块。
    """
    return LR0005(**v).get_bsrn_format()


def lr0006_format(v):
    """
    Format LR0006 ozone instrumentation metadata (R ``lr0006GetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0006`` (see ``specs.LR_SPECS``).
        ``LR0006`` 字段字典（见 ``specs.LR_SPECS``）。

    Returns
    -------
    str
        BSRN-formatted LR0006 text block.
        BSRN 格式的 LR0006 文本块。
    """
    return LR0006(**v).get_bsrn_format()


def lr0007_format(v, synop=None):
    """
    Format LR0007 station history / methods metadata (R ``lr0007GetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR0007`` (see ``specs.LR_SPECS``).
        ``LR0007`` 字段字典（见 ``specs.LR_SPECS``）。
    synop : str or None, optional
        If not ``None``, first flag in the trailing flag line is ``Y`` (SYNOP linkage).
        非 ``None`` 时尾部标志行首字符为 ``Y``（与 SYNOP 关联）。

    Returns
    -------
    str
        BSRN-formatted LR0007 text block.
        BSRN 格式的 LR0007 文本块。
    """
    return LR0007(**v).get_bsrn_format(synop=synop)


def lr0008_format(v, anyChange=False, printLr=False, LR0009Format=False):
    """
    Format LR0008 radiation instrument metadata (R ``lr0008GetBsrnFormat``).

    When ``LR0009Format`` is True, emits the compact LR0009-style layout used inside
    stacked records.

    Parameters
    ----------
    v : dict
        Field values for ``LR0008`` (see ``specs.LR_SPECS``).
        ``LR0008`` 字段字典（见 ``specs.LR_SPECS``）。
    anyChange : bool, optional
        If ``printLr`` is True, selects ``*C0008`` vs ``*U0008`` (or ``*C0009`` / ``*U0009``).
        当 ``printLr`` 为 True 时，与 ``LR0009Format`` 一起决定 ``*C0008``/``*U0008`` 或 0009。
    printLr : bool, optional
        If True, prefix the block with the ``*C``/``*U`` header line.
        为 True 时在块前加 ``*C``/``*U`` 头行。
    LR0009Format : bool, optional
        If True, use the short LR0009-style field layout instead of full LR0008.
        为 True 时使用 LR0009 式短布局而非完整 LR0008。

    Returns
    -------
    str
        BSRN-formatted LR0008 (or LR0009-style) text block.
        BSRN 格式的 LR0008（或 LR0009 式）文本块。
    """
    return LR0008(**v).get_bsrn_format(
        anyChange=anyChange, printLr=printLr, LR0009Format=LR0009Format
    )


def lr4000const_format(v, method=1):
    """
    Format ``@LR4000CONST`` calibration certificate line(s) (R ``lr4000constGetBsrnFormat``).

    Parameters
    ----------
    v : dict
        Field values for ``LR4000CONST`` (see ``specs.LR_SPECS``).
        ``LR4000CONST`` 字段字典（见 ``specs.LR_SPECS``）。
    method : int, optional
        ``1`` use ``certificateCodeID`` as-is; ``2`` build id from calibration metadata.
        ``1`` 直接使用 ``certificateCodeID``；``2`` 由校准元数据拼接证书编号。

    Returns
    -------
    str
        Wrapped ``@LR4000CONST`` line(s) for the archive.
        存档用 ``@LR4000CONST`` 折行文本。
    """
    return LR4000CONST(**v).get_bsrn_format(method=method)


# =============================================================================
# ``0_bsrnFormats_datas.R`` — LR0100, LR4000
# R ``0_bsrnFormats_datas.R``：LR0100、LR4000 数据记录
# =============================================================================


def lr0100_data_format(df, changed, yearMonth):
    """
    Format LR0100 minute-resolution data from a columnar ``DataFrame`` (R ``lr0100GetBsrnFormat``).

    Parameters
    ----------
    df : pandas.DataFrame
        One column per LR0100 field (names match ``specs.LR_SPECS`` / on-disk identifiers).
        各列对应 LR0100 字段（列名同 ``specs.LR_SPECS`` / 磁盘标识）。
    changed : bool
        If True, emit ``*C0100``; else ``*U0100``.
        为 True 输出 ``*C0100``，否则 ``*U0100``。
    yearMonth : str
        ``'YYYY-MM'`` for the month covered by ``df``.
        与 ``df`` 对应的 ``'YYYY-MM'``。

    Returns
    -------
    str
        BSRN-formatted LR0100 data block.
        BSRN 格式的 LR0100 数据块。
    """
    # Column dict plus yearMonth for LR0100. / 列字典并附加 yearMonth。
    kwargs = {c: df[c] for c in df.columns}
    kwargs["yearMonth"] = yearMonth
    return LR0100(**kwargs).get_bsrn_format(changed=changed)


def lr4000_data_format(df, changed, yearMonth):
    """
    Format LR4000 minute-resolution data from a columnar ``DataFrame`` (R ``lr4000GetBsrnFormat``).

    Parameters
    ----------
    df : pandas.DataFrame
        One column per LR4000 field (names match ``specs.LR_SPECS`` / on-disk identifiers).
        各列对应 LR4000 字段（列名同 ``specs.LR_SPECS`` / 磁盘标识）。
    changed : bool
        If True, emit ``*C4000``; else ``*U4000``.
        为 True 输出 ``*C4000``，否则 ``*U4000``。
    yearMonth : str
        ``'YYYY-MM'`` for the month covered by ``df``.
        与 ``df`` 对应的 ``'YYYY-MM'``。

    Returns
    -------
    str
        BSRN-formatted LR4000 data block.
        BSRN 格式的 LR4000 数据块。
    """
    # Column dict plus yearMonth for LR4000. / 列字典并附加 yearMonth。
    kwargs = {c: df[c] for c in df.columns}
    kwargs["yearMonth"] = yearMonth
    return LR4000(**kwargs).get_bsrn_format(changed=changed)
