"""
bsrn.archive.specs

Logical-record field definitions (conceptually R ``A1_formats`` / ``getParams``)
and static tables from R ``0_data.R``: ``LR_SPECS`` (formats), ``STATION_METADATA``
(A2), ``TOPOGRAPHIES`` / ``SURFACES`` / ``QUANTITIES`` / ``PYRGEOMETER_*`` (A3–A7).

逻辑记录字段定义（对应 R 的 ``A1_formats`` / ``getParams``）及 ``0_data.R`` 静态表：
``LR_SPECS``（格式）、``STATION_METADATA``（A2）、以及 A3–A7 分类编码表。

Format Codes (Fortran-style):
- 'I#': Integer padded to # characters (e.g., I4)
- 'A#': String/Alphanumeric padded to # characters (e.g., A38)
- 'F#.#': Float padded to total width with specific decimals (e.g., F5.1)
- 'L': Logical boolean (usually T/F or 1/0)
- 'ND': Not defined / No specific width enforced by the original spec

定义了所有 BSRN 逻辑记录的严格 ASCII 固定宽度格式化规则、缺失值、
必填字段、默认值和说明性标签。

格式代码（Fortran 风格）：
- 'I#'：填充到 # 个字符的整数（如 I4）
- 'A#'：填充到 # 个字符的字符串/字母数字（如 A38）
- 'F#.#'：填充到指定总宽度并带特定小数位的浮点数（如 F5.1）
- 'L'：逻辑布尔值（通常为 T/F 或 1/0）
- 'ND'：未定义 / 原始规范未强制要求特定宽度
"""

# LR_SPECS: per logical record (LR0001 … LR4000CONST), each field’s format, missing value,
# mandatory flag, default, and validate_func name (R ``A1_formats`` / ``getParams``).
# 各逻辑记录的字段规范：Fortran 风格格式、缺失值、必填、默认值、校验函数名（R ``A1_formats`` / ``getParams``）。
LR_SPECS = {
    "LR0001": {
        "stationNumber": {"label": "Station identification number", "format": "I2", "missing": None, "mandatory": True, "default": None, "validate_func": "I2_validateFunction"},
        "month":         {"label": "Month of measurement", "format": "I2", "missing": None, "mandatory": True, "default": None, "validate_func": "month_validateFunction"},
        "year":          {"label": "Year of measurement", "format": "I4", "missing": None, "mandatory": True, "default": None, "validate_func": "year_validateFunction"},
        "version":       {"label": "Version of data", "format": "I2", "missing": None, "mandatory": True, "default": None, "validate_func": "I2_validateFunction"}
    },
    "LR0002": {
        "scientistChange":       {"label": "Is scientist changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "scientistChangeDay":    {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "scientistChangeHour":   {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "scientistChangeMinute": {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "scientistName":         {"label": "Name", "format": "A38", "missing": None, "mandatory": True, "default": None, "validate_func": "A38_validateFunction"},
        "scientistTel":          {"label": "Telephone no.", "format": "A20", "missing": None, "mandatory": True, "default": None, "validate_func": "telephone_validateFunction"},
        "scientistFax":          {"label": "Fax no.", "format": "A20", "missing": None, "mandatory": True, "default": None, "validate_func": "telephone_validateFunction"},
        "scientistTcpip":        {"label": "TCP/IP no.", "format": "A15", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "tcpip_validateFunction"},
        "scientistMail":         {"label": "E-mail address", "format": "A50", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "email_validateFunction"},
        "scientistAddress":      {"label": "Address", "format": "A80", "missing": None, "mandatory": True, "default": None, "validate_func": "A80_validateFunction"},
        "deputyChange":          {"label": "Is deputy changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "deputyChangeDay":       {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "deputyChangeHour":      {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "deputyChangeMinute":    {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "deputyName":            {"label": "Name", "format": "A38", "missing": None, "mandatory": True, "default": None, "validate_func": "A38_validateFunction"},
        "deputyTel":             {"label": "Telephone no.", "format": "A20", "missing": None, "mandatory": True, "default": None, "validate_func": "telephone_validateFunction"},
        "deputyFax":             {"label": "Fax no.", "format": "A20", "missing": None, "mandatory": True, "default": None, "validate_func": "telephone_validateFunction"},
        "deputyTcpip":           {"label": "TCP/IP no.", "format": "A15", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "tcpip_validateFunction"},
        "deputyMail":            {"label": "E-mail address", "format": "A50", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "email_validateFunction"},
        "deputyAddress":         {"label": "Address", "format": "A80", "missing": None, "mandatory": True, "default": None, "validate_func": "A80_validateFunction"}
    },
    "LR0003": {
        "message": {"label": "Messages not to be inserted in the BSRN database", "format": "A", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A_validateFunction"}
    },
    "LR0004": {
        "stationDescChange":       {"label": "Is station description changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "stationDescChangeDay":    {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "stationDescChangeHour":   {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "stationDescChangeMinute": {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "surfaceType":             {"label": "Surface type", "format": "I2", "missing": None, "mandatory": True, "default": None, "validate_func": "surface_validateFunction"},
        "topographyType":          {"label": "Topography Type", "format": "I2", "missing": None, "mandatory": True, "default": None, "validate_func": "topography_validateFunction"},
        "address":                 {"label": "Address", "format": "A80", "missing": None, "mandatory": True, "default": None, "validate_func": "A80_validateFunction"},
        "telephone":               {"label": "Telephone no. of station", "format": "A20", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "telephone_validateFunction"},
        "fax":                     {"label": "Fax no. of station", "format": "A20", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "telephone_validateFunction"},
        "tcpip":                   {"label": "TCP/IP no. of station", "format": "A15", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "tcpip_validateFunction"},
        "mail":                    {"label": "E-mail address of station", "format": "A50", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "email_validateFunction"},
        "latitude":                {"label": "Latitude [degrees, 0 is Southpole, positive is northward]", "format": "F7.3", "missing": None, "mandatory": True, "default": None, "validate_func": "latitude_validateFunction"},
        "longitude":               {"label": "Longitude [degrees, 0 is 180W, positive is eastwards]", "format": "F7.3", "missing": None, "mandatory": True, "default": None, "validate_func": "longitude_validateFunction"},
        "altitude":                {"label": "Altitude [m above the sea]", "format": "I4", "missing": None, "mandatory": True, "default": None, "validate_func": "I4_validateFunction"},
        "synop":                   {"label": "Identification od 'SYNOP' station", "format": "A5", "missing": "XXXXX", "mandatory": False, "default": None, "validate_func": "A5_validateFunction"},
        "horizonChange":           {"label": "Is horizon changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "horizonChangeDay":        {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "horizonChangeHour":       {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "horizonChangeMinute":     {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "azimuth":                 {"label": "Azimuth [degrees from north clockwise]; format : 'A1,A2,...,An'", "format": "ND", "missing": -1, "mandatory": False, "default": None, "validate_func": "azimuth_validateFunction"},
        "elevation":               {"label": "Elevation [degrees]; format : 'E1,E2,...,En'", "format": "ND", "missing": -1, "mandatory": False, "default": None, "validate_func": "elevation_validateFunction"}
    },
    "LR0005": {
        "change":           {"label": "Is radiosonde equipment changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "changeDay":        {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "changeHour":       {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "changeMinute":     {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "operating":        {"label": "Is radiosonde operating?", "format": "A1", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "manufacturer":     {"label": "Manufacturer", "format": "A30", "missing": None, "mandatory": True, "default": None, "validate_func": "A30_validateFunction"},
        "location":         {"label": "Location", "format": "A25", "missing": None, "mandatory": True, "default": None, "validate_func": "A25_validateFunction"},
        "distanceFromSite": {"label": "Distance from radiation site [km]", "format": "I3", "missing": None, "mandatory": True, "default": None, "validate_func": "I3_validateFunction"},
        "time1stLaunch":    {"label": "Time of 1st launch [h UTC]", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "time2ndLaunch":    {"label": "Time of 2nd launch [h UTC]", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "time3rdLaunch":    {"label": "Time of 3rd launch [h UTC]", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "time4thLaunch":    {"label": "Time of 4th launch [h UTC]", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "identification":   {"label": "Identification of radiosonde", "format": "A5", "missing": None, "mandatory": True, "default": None, "validate_func": "A5_validateFunction"},
        "remarks":          {"label": "Remarks about radiosonde", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"}
    },
    "LR0006": {
        "change":           {"label": "Is ozone measurements equipment changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "changeDay":        {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "changeHour":       {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "changeMinute":     {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "operating":        {"label": "Is ozone measurements opérating ?", "format": "A1", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "manufacturer":     {"label": "Manufacturer", "format": "A30", "missing": None, "mandatory": True, "default": None, "validate_func": "A30_validateFunction"},
        "location":         {"label": "Location", "format": "A25", "missing": None, "mandatory": True, "default": None, "validate_func": "A25_validateFunction"},
        "distanceFromSite": {"label": "Distance from radiation site [km]", "format": "I3", "missing": None, "mandatory": True, "default": None, "validate_func": "I3_validateFunction"},
        "identification":   {"label": "Identification number of ozone instrument", "format": "A5", "missing": None, "mandatory": True, "default": None, "validate_func": "A5_validateFunction"},
        "remarks":          {"label": "Remarks about ozone measurements", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"}
    },
    "LR0007": {
        "change":          {"label": "Is station history changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "changeDay":       {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "changeHour":      {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "changeMinute":    {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "cloudAmount":     {"label": "Method est. cloud amount (digital proc.)", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"},
        "cloudBaseHeight": {"label": "Method est. cloud base height (with instrument)", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"},
        "cloudLiquid":     {"label": "Method est. cloud liquid water content", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"},
        "cloudAerosol":    {"label": "Method est. cloud aerosol vertical distribution", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"},
        "waterVapour":     {"label": "Method est. water vapour press. v.d. (A80)", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"}
    },
    "LR0008": {
        "change":                    {"label": "Is instruments changed this month ?", "format": "L", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "changeDay":                 {"label": "Day", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "day_validateFunction"},
        "changeHour":                {"label": "Hour", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "hour_validateFunction"},
        "changeMinute":              {"label": "Minute", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "minute_validateFunction"},
        "operating":                 {"label": "Is instrument measuring?", "format": "A1", "missing": None, "mandatory": True, "default": False, "validate_func": "L_validateFunction"},
        "radiationQuantityMeasured": {"label": "Radiation quantity measured", "format": "I9", "missing": None, "mandatory": True, "default": None, "validate_func": "quantities_validateFunction"},
        "manufacturer":              {"label": "Manufacturer", "format": "A30", "missing": None, "mandatory": True, "default": None, "validate_func": "A30_validateFunction"},
        "model":                     {"label": "Model", "format": "A15", "missing": None, "mandatory": True, "default": None, "validate_func": "A15_validateFunction"},
        "serialNumber":              {"label": "Serial number", "format": "A18", "missing": None, "mandatory": True, "default": None, "validate_func": "A18_validateFunction"},
        "dateOfPurchase":            {"label": "Date of Purchase", "format": "A8", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "date_validateFunction"},
        "identification":            {"label": "Identification number assigned by the WRMC", "format": "I5", "missing": None, "mandatory": True, "default": None, "validate_func": "I5_validateFunction"},
        "remarks":                   {"label": "Remarks about the radiation instrument", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"},
        "pyrgeometerBody":           {"label": "Pyrgeometer body compensation code", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "body_validateFunction"},
        "pyrgeometerDome":           {"label": "Pyrgeometer dome compensation code", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "dome_validateFunction"},
        "numOfBand":                 {"label": "Number of band (for spectral instruments)", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "I2_validateFunction"},
        "wavelenghBand1":            {"label": "Wavelength of band 1 of spectral i. [micron]", "format": "F7.3", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F7.3_validateFunction"},
        "bandwidthBand1":            {"label": "Bandwidth of band 1 of spectral i. [micron]", "format": "F7.3", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F7.3_validateFunction"},
        "wavelenghBand2":            {"label": "Wavelength of band 2", "format": "F7.3", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F7.3_validateFunction"},
        "bandwidthBand2":            {"label": "Bandwidth of band 2", "format": "F7.3", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F7.3_validateFunction"},
        "wavelenghBand3":            {"label": "Wavelength of band 3", "format": "F7.3", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F7.3_validateFunction"},
        "bandwidthBand3":            {"label": "Bandwidth of band 3", "format": "F7.3", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F7.3_validateFunction"},
        "maxZenithAngle":            {"label": "Max. zenith angle [degree] of direct", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "zenith_validateFunction"},
        "minSpectral":               {"label": "Min. (spectral) instrument", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "zenith_validateFunction"},
        "location":                  {"label": "Location of calibration", "format": "A30", "missing": None, "mandatory": True, "default": None, "validate_func": "A30_validateFunction"},
        "person":                    {"label": "Person doing calibration", "format": "A40", "missing": None, "mandatory": True, "default": None, "validate_func": "A40_validateFunction"},
        "startOfCalibPeriod1":       {"label": "Start of calibration period (band 1 of spectr. instr.)", "format": "A8", "missing": None, "mandatory": True, "default": None, "validate_func": "date_validateFunction"},
        "endOfCalibPeriod1":         {"label": "End of ... (both [MM/DD/YY])", "format": "A8", "missing": None, "mandatory": True, "default": None, "validate_func": "date_validateFunction"},
        "numOfComp1":                {"label": "Number of comparisons (band 1 of spectr. instr.)", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "I2_validateFunction"},
        "meanCalibCoeff1":           {"label": "Mean calibration coefficient (band 1 of spectr. instr.)", "format": "F12.4", "missing": None, "mandatory": True, "default": None, "validate_func": "F12.4_validateFunction"},
        "stdErrorCalibCoeff1":       {"label": "Standard error of cal. coeff. (band 1 of spectr. instr.)", "format": "F12.4", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F12.4_validateFunction"},
        "startOfCalibPeriod2":       {"label": "Start of calibration period (band 2 of spectr. instr.)", "format": "A8", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "date_validateFunction"},
        "endOfCalibPeriod2":         {"label": "End of ... (both [MM/DD/YY])", "format": "A8", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "date_validateFunction"},
        "numOfComp2":                {"label": "Number of comparisons (band 2 of spectr. instr.)", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "I2_validateFunction"},
        "meanCalibCoeff2":           {"label": "Mean calibration coefficient (band 2 of spectr. instr.)", "format": "F12.4", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F12.4_validateFunction"},
        "stdErrorCalibCoeff2":       {"label": "Standard error of cal. coeff. (band 2 of spectr. instr.)", "format": "F12.4", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F12.4_validateFunction"},
        "startOfCalibPeriod3":       {"label": "Start of calibration period (band 3 of spectr. instr.)", "format": "A8", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "date_validateFunction"},
        "endOfCalibPeriod3":         {"label": "End of ... (both [MM/DD/YY])", "format": "A8", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "date_validateFunction"},
        "numOfComp3":                {"label": "Number of comparisons (band 3 of spectr. instr.)", "format": "I2", "missing": -1, "mandatory": False, "default": None, "validate_func": "I2_validateFunction"},
        "meanCalibCoeff3":           {"label": "Mean calibration coefficient (band 3 of spectr. instr.)", "format": "F12.4", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F12.4_validateFunction"},
        "stdErrorCalibCoeff3":       {"label": "Standard error of cal. coeff. (band 3 of spectr. instr.)", "format": "F12.4", "missing": -1.0, "mandatory": False, "default": None, "validate_func": "F12.4_validateFunction"},
        "remarksOnCalib1":           {"label": "Remarks on calibration, e.g. units of cal. coeff.", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"},
        "remarksOnCalib2":           {"label": "Remarks on calibration (continued)", "format": "A80", "missing": "XXX", "mandatory": False, "default": None, "validate_func": "A80_validateFunction"}
    },
    "LR0100": {
        "yearMonth":   {"label": "Year and month of measurement ('YYYY-MM')", "format": "A7", "missing": None, "mandatory": True, "default": None, "validate_func": "genericValidateFunction"},
        "ghi_avg":     {"label": "Global mean", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "ghi_std":     {"label": "Global standard deviation", "format": "F5.1", "missing": -99.9, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "ghi_min":     {"label": "Global minimum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "ghi_max":     {"label": "Global maximum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "bni_avg":     {"label": "Direct mean", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "bni_std":     {"label": "Direct standard deviation", "format": "F5.1", "missing": -99.9, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "bni_min":     {"label": "Direct minimum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "bni_max":     {"label": "Direct maximum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "dhi_avg":     {"label": "Diffuse mean", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "dhi_std":     {"label": "Diffuse standard deviation", "format": "F5.1", "missing": -99.9, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "dhi_min":     {"label": "Diffuse minimum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "dhi_max":     {"label": "Diffuse maximum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "lwd_avg":     {"label": "Downward long-wave radiation mean", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "lwd_std":     {"label": "Downward long-wave radiation standard deviation", "format": "F5.1", "missing": -99.9, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "lwd_min":     {"label": "Downward long-wave radiation minimum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "lwd_max":     {"label": "Downward long-wave radiation maximum", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "temperature": {"label": "Air temperature at downward long-wave instrument height", "format": "F5.1", "missing": -99.9, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "humidity":    {"label": "Relative humidity at downward long-wave instrument height", "format": "F5.1", "missing": -99.9, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"},
        "pressure":    {"label": "Pressure at downward long-wave instrument height", "format": "I4", "missing": -999, "mandatory": False, "default": None, "validate_func": "LR0100_validateFunction"}
    },
    "LR4000": {
        "yearMonth":     {"label": "Year and month of measurement ('YYYY-MM')", "format": "A7", "missing": None, "mandatory": True, "default": None, "validate_func": "genericValidateFunction"},
        "domeT1_down":   {"label": "dome temperature 1downward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "domeT2_down":   {"label": "dome temperature 2 downward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "domeT3_down":   {"label": "dome temperature 3 downward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "bodyT_down":    {"label": "body temperature downward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "longwave_down": {"label": "thermopile output downward long-wave instrument", "format": "F6.1", "missing": -999.9, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "domeT1_up":     {"label": "dome temperature 1upward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "domeT2_up":     {"label": "dome temperature 2 upward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "domeT3_up":     {"label": "dome temperature 3 upward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "bodyT_up":      {"label": "body temperature upward long-wave instrument", "format": "F6.2", "missing": -99.99, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"},
        "longwave_up":   {"label": "thermopile output upward long-wave instrument", "format": "F6.1", "missing": -999.9, "mandatory": False, "default": None, "validate_func": "LR4000_validateFunction"}
    },
    "LR4000CONST": {
        "serialNumber_Manufacturer": {"label": "The serial number as it appears in the calibration certificate/instrument plate", "format": "I6", "missing": None, "mandatory": True, "default": None, "validate_func": "A_validateFunction"},
        "serialNumber_WRMC":         {"label": "The serial nimber used in your station-to-archive files (LR0008/0009) to identify the instrument with this serial number (e.g. for dom: \"74xxx\", with xxx=001,002,003,...)", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "A_validateFunction"},
        "certificateCodeID":         {"label": "The station scientist can define CertificateCodeId according to one of the 2 options. See LR4000_TableA1Update2", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "A_validateFunction"},
        "yyyymmdd":                  {"label": "date of the calibration certificate issued : numeric format yyyymmdd", "format": "I8", "missing": None, "mandatory": False, "default": None, "validate_func": "I8_validateFunction"},
        "manufact":                  {"label": "KZ (Kipp and Zonen), EP (Eppley), HF (Hukseflux),...", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "A_validateFunction"},
        "model":                     {"label": "CH1,CH1P,CM11,CM21,CM21P,CM22,CM22P,CG4,CGR4,...", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "A_validateFunction"},
        "C":                         {"label": "Thermopile responsivity", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "C_validateFunction"},
        "k0":                        {"label": "General equation of the pyrgeometer : ki are the instrument dependent calibration constants", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "C_validateFunction"},
        "k1":                        {"label": "General equation of the pyrgeometer : ki are the instrument dependent calibration constants", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "C_validateFunction"},
        "k2":                        {"label": "General equation of the pyrgeometer : ki are the instrument dependent calibration constants", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "C_validateFunction"},
        "k3":                        {"label": "General equation of the pyrgeometer : ki are the instrument dependent calibration constants", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "C_validateFunction"},
        "f":                         {"label": "General equation of the pyrgeometer : correction factor for infrared irradiance on unshaded domes", "format": "ND", "missing": None, "mandatory": False, "default": None, "validate_func": "C_validateFunction"}
    }
}

# STATION_METADATA: BSRN station directory by 3-letter code (name, lat/lon, contacts, …).
# R ``A2_bsrnStations`` in ``0_data.R``. Used when filling LR0001 / LR0002 / LR0004 style fields.
# BSRN 站点目录（三位代码 → 名称、坐标、联系人等）；用于 LR0001 / LR0002 / LR0004 等字段。
STATION_METADATA = {
    "ABS": {"name": "Abashiri", "location": "Hokkaido, Japan", "station_no": None, "lat": 44.0178, "lon": 144.2797, "elevation": 38.0, "established": "2021-03-01", "closed": None, "surface_type": "asphalt", "topography_type": "flat, rural", "scientist": "Sasaki Shun", "email": "rrc-jma@met.kishou.go.jp", "url": None},
    "ALE": {"name": "Alert", "location": "Lincoln Sea", "station_no": 18, "lat": 82.4900, "lon": -62.4200, "elevation": 127.0, "established": "2004-08-16", "closed": None, "surface_type": "tundra", "topography_type": "hilly, rural", "scientist": "Christopher Cox", "email": "christopher.j.cox@noaa.gov", "url": None},
    "ASP": {"name": "Alice Springs", "location": "Macdonnell Ranges, Northern Territory, Australia", "station_no": 1, "lat": -23.7980, "lon": 133.8880, "elevation": 547.0, "established": "1995-01-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Matt Tully", "email": "m.tully@bom.gov.au", "url": None},
    "BAR": {"name": "Barrow", "location": "Alaska, USA", "station_no": 22, "lat": 71.3230, "lon": -156.6070, "elevation": 8.0, "established": "1992-01-01", "closed": None, "surface_type": "tundra", "topography_type": "flat, rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": "http://www.esrl.noaa.gov/gmd/obop/brw/"},
    "BER": {"name": "Bermuda", "location": "Bermuda", "station_no": 24, "lat": 32.2670, "lon": -64.6670, "elevation": 8.0, "established": "1992-01-01", "closed": None, "surface_type": "water, ocean", "topography_type": "flat, rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": "http://www.esrl.noaa.gov/gmd/dv/site/BPH.html"},
    "BIL": {"name": "Billings", "location": "Oklahoma, United States of America", "station_no": 28, "lat": 36.6050, "lon": -97.5160, "elevation": 317.0, "established": "1993-06-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": None},
    "BON": {"name": "Bondville", "location": "Illinois, United States of America", "station_no": 32, "lat": 40.0667, "lon": -88.3667, "elevation": 213.0, "established": "1995-01-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": "http://www.srrb.noaa.gov/surfrad/bondvill.html"},
    "BOS": {"name": "Boulder", "location": "Colorado, United States of America", "station_no": 34, "lat": 40.1250, "lon": -105.2370, "elevation": 1689.0, "established": "1995-07-01", "closed": None, "surface_type": "grass", "topography_type": "hilly, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": "http://www.srrb.noaa.gov/surfrad/tablemt.html"},
    "BOU": {"name": "Boulder", "location": "Colorado, United States of America", "station_no": 23, "lat": 40.0500, "lon": -105.0070, "elevation": 1577.0, "established": "1992-01-01", "closed": "2016-06-30", "surface_type": "grass", "topography_type": "flat, rural", "scientist": "David Longenecker", "email": "David.U.Longenecker@noaa.gov", "url": None},
    "BRB": {"name": "Brasilia", "location": "Brasilia City, Distrito Federal, Brazil", "station_no": 71, "lat": -15.6010, "lon": -47.7130, "elevation": 1023.0, "established": "2006-02-01", "closed": None, "surface_type": "concrete, since 2015: shrub", "topography_type": "flat, rural", "scientist": "Enio Bueno Pereira", "email": "enio.pereira@inpe.br", "url": None},
    "BUD": {"name": "Budapest-Lorinc", "location": "Hungary, Budapest", "station_no": 14, "lat": 47.4291, "lon": 19.1822, "elevation": 139.1, "established": None, "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Dénes Fekete", "email": "fekete.d@met.hu", "url": None},
    "CAB": {"name": "Cabauw", "location": "The Netherlands", "station_no": 53, "lat": 51.9711, "lon": 4.9267, "elevation": 0.0, "established": "2005-12-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Wouter Knap", "email": "knap@knmi.nl", "url": "http://www.knmi.nl/bsrn/"},
    "CAM": {"name": "Camborne", "location": "United Kingdom", "station_no": 50, "lat": 50.2167, "lon": -5.3167, "elevation": 88.0, "established": "2001-01-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Fraser Cunningham", "email": "fraser.cunningham@metoffice.gov.uk", "url": "http://hdl.handle.net/10013/epic.47488.d001"},
    "CAP": {"name": "Cape Baranova", "location": "Russia", "station_no": 15, "lat": 79.2700, "lon": 101.7500, "elevation": None, "established": "2016-01-01", "closed": None, "surface_type": "tundra", "topography_type": "flat, rural", "scientist": "Vasilii Kustov", "email": "kustov@aari.nw.ru", "url": "http://www.aari.ru/main.php?lg=1"},
    "CAR": {"name": "Carpentras", "location": "France", "station_no": 10, "lat": 44.0830, "lon": 5.0590, "elevation": 100.0, "established": "1996-08-01", "closed": "2018-12-31", "surface_type": "cultivated", "topography_type": "hilly, rural", "scientist": "Thierry Duprat", "email": "thierry.duprat@meteo.fr", "url": None},
    "CLH": {"name": "Chesapeake Light", "location": "North Atlantic Ocean", "station_no": 39, "lat": 36.9050, "lon": -75.7130, "elevation": 37.0, "established": "2000-06-01", "closed": "2017-12-31", "surface_type": "water, ocean", "topography_type": "flat, rural", "scientist": "Fred M. Denn", "email": "Frederick.M.Denn@nasa.gov", "url": "http://cove.larc.nasa.gov"},
    "CNR": {"name": "Cener", "location": "Spain, Sarriguren, Navarra", "station_no": 45, "lat": 42.8160, "lon": -1.6010, "elevation": 471.0, "established": "2009-07-01", "closed": None, "surface_type": "asphalt", "topography_type": "mountain valley, urban", "scientist": "Xabier Olano", "email": "xolano@cener.com", "url": "http://www.cener.com"},
    "COC": {"name": "Cocos Island", "location": "Cocos (Keeling) Islands", "station_no": 47, "lat": -12.1930, "lon": 96.8350, "elevation": 6.0, "established": "2004-09-14", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Matt Tully", "email": "m.tully@bom.gov.au", "url": None},
    "DAA": {"name": "De Aar", "location": "South Africa", "station_no": 40, "lat": -30.6667, "lon": 23.9930, "elevation": 1287.0, "established": "2000-05-01", "closed": None, "surface_type": "sand", "topography_type": "flat, rural", "scientist": "Lucky Ntsangwane", "email": "lucky.ntsangwane@weathersa.co.za", "url": None},
    "DAR": {"name": "Darwin", "location": "Australia", "station_no": 2, "lat": -12.4250, "lon": 130.8910, "elevation": 30.0, "established": "2002-06-01", "closed": "2015-01-31", "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Charles Long", "email": None, "url": None},
    "DOM": {"name": "Concordia Station, Dome C", "location": "Antarctica", "station_no": 74, "lat": -75.1000, "lon": 123.3830, "elevation": 3233.0, "established": "2006-01-01", "closed": None, "surface_type": "glacier, accumulation area", "topography_type": "flat, rural", "scientist": "Angelo Lupi", "email": "a.lupi@isac.cnr.it", "url": "http://www.italiantartide.it/?page_id=272"},
    "DRA": {"name": "Desert Rock", "location": "Nevada, United States of America", "station_no": 35, "lat": 36.6260, "lon": -116.0180, "elevation": 1007.0, "established": "1998-02-01", "closed": None, "surface_type": "desert, gravel", "topography_type": "flat, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": "http://www.srrb.noaa.gov/surfrad/desrock.html"},
    "DWN": {"name": "Darwin Met Office", "location": "Australia", "station_no": 65, "lat": -12.4240, "lon": 130.8925, "elevation": 32.0, "established": "2008-04-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Matt Tully", "email": "m.tully@bom.gov.au", "url": None},
    "E13": {"name": "Southern Great Plains", "location": "Oklahoma, United States of America", "station_no": 27, "lat": 36.6050, "lon": -97.4850, "elevation": 318.0, "established": "1997-08-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": "http://www.arm.gov/sites/sgp"},
    "ENA": {"name": "Eastern North Atlantic", "location": "Azores", "station_no": 77, "lat": 39.0911, "lon": -28.0292, "elevation": 15.2, "established": "2013-09-28", "closed": None, "surface_type": "grass", "topography_type": "hilly/rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": None},
    "EUR": {"name": "Eureka", "location": "Ellesmere Island, Canadian Arctic Archipelago", "station_no": 19, "lat": 79.9890, "lon": -85.9404, "elevation": 85.0, "established": "2007-09-01", "closed": "2011-12-31", "surface_type": "tundra", "topography_type": "hilly, rural", "scientist": None, "email": None, "url": "https://en.wikipedia.org/wiki/Eureka,_Nunavut"},
    "FLO": {"name": "Florianopolis", "location": "South Atlantic Ocean", "station_no": 3, "lat": -27.6047, "lon": -48.5227, "elevation": 11.0, "established": "1994-06-01", "closed": None, "surface_type": "concrete", "topography_type": "mountain valley, urban", "scientist": "Sergio Colle", "email": "sergio.colle@ufsc.br", "url": None},
    "FPE": {"name": "Fort Peck", "location": "Montana, United States of America", "station_no": 31, "lat": 48.3167, "lon": -105.1000, "elevation": 634.0, "established": "1995-01-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": "http://www.srrb.noaa.gov/surfrad/ftpeck.html"},
    "FUA": {"name": "Fukuoka", "location": "Japan", "station_no": 6, "lat": 33.5822, "lon": 130.3764, "elevation": 3.0, "established": "2010-04-01", "closed": None, "surface_type": "asphalt", "topography_type": "flat, urban", "scientist": "Kouichi Nakashima", "email": "rrc-jma@met.kishou.go.jp", "url": "http://hdl.handle.net/10013/epic.44630.d041"},
    "GAN": {"name": "Gandhinagar", "location": "India", "station_no": 58, "lat": 23.1101, "lon": 72.6276, "elevation": 65.0, "established": "2014-05-19", "closed": None, "surface_type": "shrub", "topography_type": "flat, urban", "scientist": "Prasun Kumar", "email": "prasun.niwe@nic.in", "url": None},
    "GCR": {"name": "Goodwin Creek", "location": "Mississippi, United States of America", "station_no": 33, "lat": 34.2547, "lon": -89.8729, "elevation": 98.0, "established": "1995-01-01", "closed": None, "surface_type": "grass", "topography_type": "hilly, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": "http://www.srrb.noaa.gov/surfrad/goodwin.html"},
    "GIM": {"name": "Granite Island", "location": "Michigan, United States", "station_no": 78, "lat": 46.7210, "lon": -87.4110, "elevation": 208.0, "established": None, "closed": None, "surface_type": "grass", "topography_type": "flat, urban", "scientist": "Bryan Fabbri", "email": "bryan.e.fabbri@nasa.gov", "url": None},
    "GOB": {"name": "Gobabeb", "location": "Namib Desert, Namibia", "station_no": 20, "lat": -23.5614, "lon": 15.0420, "elevation": 407.0, "established": "2012-05-15", "closed": None, "surface_type": "desert gravel", "topography_type": "flat rural", "scientist": "Roland Vogt", "email": "Roland.Vogt@unibas.ch", "url": None},
    "GUR": {"name": "Gurgaon", "location": "India", "station_no": 56, "lat": 28.4249, "lon": 77.1560, "elevation": 259.0, "established": "2014-04-21", "closed": None, "surface_type": "shrub", "topography_type": "flat, urban", "scientist": "Prasun Kumar", "email": "prasun.niwe@nic.in", "url": None},
    "GVN": {"name": "Georg von Neumayer", "location": "Dronning Maud Land, Antarctica", "station_no": 13, "lat": -70.6500, "lon": -8.2500, "elevation": 42.0, "established": "1992-01-01", "closed": None, "surface_type": "iceshelf", "topography_type": "flat, rural", "scientist": "Holger Schmithüsen", "email": "Holger.Schmithuesen@awi.de", "url": "http://www.awi.de/en/science/long-term-observations/atmosphere/antarctic-neumayer.html"},
    "HOW": {"name": "Howrah", "location": "India", "station_no": 57, "lat": 22.5535, "lon": 88.3064, "elevation": 51.0, "established": "2014-06-15", "closed": None, "surface_type": "shrub", "topography_type": "flat, urban", "scientist": "Prasun Kumar", "email": "prasun.niwe@nic.in", "url": None},
    "ILO": {"name": "Ilorin", "location": "Nigeria", "station_no": 38, "lat": 8.5333, "lon": 4.5667, "elevation": 350.0, "established": "1992-08-01", "closed": "2005-07-31", "surface_type": "shrub", "topography_type": "flat, rural", "scientist": "T O Aro", "email": None, "url": None},
    "INO": {"name": "Marguele", "location": "Romania", "station_no": None, "lat": 44.3439, "lon": 26.0123, "elevation": 110.0, "established": "2021-05-01", "closed": None, "surface_type": "cultivated", "topography_type": "flat, rural", "scientist": "Emil Carstea", "email": "emil.carstea@inoe.ro", "url": None},
    "ISH": {"name": "Ishigakijima", "location": "Japan", "station_no": 7, "lat": 24.3367, "lon": 124.1644, "elevation": 5.7, "established": "2010-04-01", "closed": None, "surface_type": "asphalt", "topography_type": "flat, rural", "scientist": "Kouichi Nakashima", "email": "rrc-jma@met.kishou.go.jp", "url": "http://hdl.handle.net/10013/epic.44630.d026"},
    "IZA": {"name": "Izaña", "location": "Tenerife, Spain", "station_no": 61, "lat": 28.3093, "lon": -16.4993, "elevation": 2372.9, "established": "2009-03-01", "closed": None, "surface_type": "rock", "topography_type": "mountain top, rural", "scientist": "Emilio Cuevas-Agulló", "email": "ecuevasa@aemet.es", "url": "http://www.bsrn.aemet.es"},
    "KWA": {"name": "Kwajalein", "location": "North Pacific Ocean", "station_no": 25, "lat": 8.7200, "lon": 167.7310, "elevation": 10.0, "established": "1992-03-01", "closed": None, "surface_type": "water, ocean", "topography_type": "flat, rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": "http://www.esrl.noaa.gov/gmd/grad/sites/kwa.html"},
    "LAU": {"name": "Lauder", "location": "New Zealand", "station_no": 60, "lat": -45.0450, "lon": 169.6890, "elevation": 350.0, "established": "1998-07-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Matt Tully", "email": "m.tully@bom.gov.au", "url": "http://gaw.empa.ch/gawsis/reports.asp?StationID=62"},
    "LER": {"name": "Lerwick", "location": "Shetland Island, United Kingdom", "station_no": 51, "lat": 60.1389, "lon": -1.1847, "elevation": 80.0, "established": "2001-01-01", "closed": None, "surface_type": "grass", "topography_type": "hilly, rural", "scientist": "Fraser Cunningham", "email": "fraser.cunningham@metoffice.gov.uk", "url": "http://hdl.handle.net/10013/epic.43134.d001"},
    "LIN": {"name": "Lindenberg", "location": "Germany", "station_no": 12, "lat": 52.2100, "lon": 14.1220, "elevation": 125.0, "established": "1994-09-01", "closed": None, "surface_type": "cultivated", "topography_type": "hilly, rural", "scientist": "Stefan Wacker", "email": "Stefan.Wacker@dwd.de", "url": "http://www.dwd.de/DE/forschung/atmosphaerenbeob/lindenbergersaeule/mol/mol_node.html"},
    "LMP": {"name": "Lampedusa", "location": "Italy", "station_no": None, "lat": 35.5180, "lon": 12.6300, "elevation": 50.0, "established": "2023-12-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Daniela Meloni", "email": "daniela.meloni@enea.it", "url": None},
    "LRC": {"name": "Langley Research Center", "location": "Hampton, Virginia, USA", "station_no": 49, "lat": 37.1038, "lon": -76.3872, "elevation": 3.0, "established": "2014-12-01", "closed": None, "surface_type": "grass", "topography_type": "flat, urban", "scientist": "Fred M. Denn", "email": "Frederick.M.Denn@nasa.gov", "url": "http://capable.larc.nasa.gov/"},
    "LYU": {"name": "Lanyu Island", "location": "Taiwan", "station_no": 79, "lat": 22.0370, "lon": 121.5583, "elevation": 324.0, "established": "2018-12-01", "closed": None, "surface_type": "forest mixed", "topography_type": "mountain top, rural", "scientist": "Kun-Wei Lin", "email": "adenins@cwb.gov.tw", "url": None},
    "MAN": {"name": "Momote", "location": "Papua New Guinea", "station_no": 29, "lat": -2.0580, "lon": 147.4250, "elevation": 6.0, "established": "1996-09-01", "closed": "2013-10-31", "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Charles Long", "email": None, "url": None},
    "MNM": {"name": "Minamitorishima", "location": "Minami-Torishima", "station_no": 8, "lat": 24.2883, "lon": 153.9833, "elevation": 7.1, "established": "2010-04-01", "closed": None, "surface_type": "water (ocean)", "topography_type": "flat, rural", "scientist": "Kouichi Nakashima", "email": "rrc-jma@met.kishou.go.jp", "url": "http://hdl.handle.net/10013/epic.44630.d003"},
    "NAU": {"name": "Nauru Island", "location": "Nauru", "station_no": 30, "lat": -0.5210, "lon": 166.9167, "elevation": 7.0, "established": "1998-11-01", "closed": "2013-09-30", "surface_type": "rock", "topography_type": "flat, rural", "scientist": "Charles Long", "email": None, "url": None},
    "NEW": {"name": "Newcastle", "location": "Australia", "station_no": 52, "lat": -32.8842, "lon": 151.7289, "elevation": 18.5, "established": None, "closed": None, "surface_type": "grass", "topography_type": "hilly, urban", "scientist": "Ben Duck", "email": "Benjamin.Duck@csiro.au", "url": None},
    "NYA": {"name": "Ny-Ålesund", "location": "Ny-Ålesund, Spitsbergen", "station_no": 11, "lat": 78.9250, "lon": 11.9300, "elevation": 11.0, "established": "1992-08-01", "closed": None, "surface_type": "tundra", "topography_type": "mountain valley, rural", "scientist": "Marion Maturilli", "email": "Marion.Maturilli@awi.de", "url": "http://www.awi.de/en/infrastructure/stations/awipev_arctic_research_base/"},
    "OHY": {"name": "Observatory of Huancayo", "location": "Peru", "station_no": 80, "lat": -12.0500, "lon": -75.3200, "elevation": 3314.0, "established": "2017-08-01", "closed": None, "surface_type": "grass", "topography_type": "mountain valley, rural", "scientist": "Luis Suarez Salas", "email": "lsuarez@igp.gob.pe", "url": None},
    "PAL": {"name": "Palaiseau, SIRTA Observatory", "location": "France", "station_no": 63, "lat": 48.7130, "lon": 2.2080, "elevation": 156.0, "established": "2003-05-01", "closed": None, "surface_type": "concrete", "topography_type": "flat, urban", "scientist": "Jordi Badosa", "email": "jordi.badosa@lmd.polytechnique.fr", "url": None},
    "PAR": {"name": "Paramaribo", "location": "South America, Surinam", "station_no": 66, "lat": 5.8060, "lon": -55.2146, "elevation": 4.0, "established": "2019-01-01", "closed": None, "surface_type": "grass", "topography_type": "flat, urban", "scientist": "Ankie Piters", "email": "piters@knmi.nl", "url": None},
    "PAY": {"name": "Payerne", "location": "Switzerland", "station_no": 21, "lat": 46.8150, "lon": 6.9440, "elevation": 491.0, "established": "1992-09-01", "closed": None, "surface_type": "cultivated", "topography_type": "hilly, rural", "scientist": "Laurent Vuilleumier", "email": "laurent.vuilleumier@meteoswiss.ch", "url": None},
    "PSU": {"name": "Rock Springs", "location": "Pennsylvania, United States of America", "station_no": 36, "lat": 40.7200, "lon": -77.9333, "elevation": 376.0, "established": "1998-05-01", "closed": None, "surface_type": "cultivated", "topography_type": "mountain valley, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": None},
    "PTR": {"name": "Petrolina", "location": "Brazil", "station_no": 72, "lat": -9.0680, "lon": -40.3190, "elevation": 387.0, "established": "2006-12-01", "closed": None, "surface_type": "concrete, since 2015: shrub", "topography_type": "flat, rural", "scientist": "Enio Bueno Pereira", "email": "enio.pereira@inpe.br", "url": None},
    "QIQ": {"name": "Qiqihar", "location": "China", "station_no": 94, "lat": 47.3833, "lon": 123.9167, "elevation": 146.0, "established": "2023-01-01", "closed": None, "surface_type": "concrete", "topography_type": "flat - rural", "scientist": "Dazhi Yang", "email": "yangdazhi.nus@gmail.com", "url": None},
    "REG": {"name": "Regina", "location": "Canada", "station_no": 5, "lat": 50.2050, "lon": -104.7130, "elevation": 578.0, "established": "1995-01-01", "closed": "2011-12-31", "surface_type": "cultivated", "topography_type": "flat, rural", "scientist": None, "email": None, "url": None},
    "RLM": {"name": "Rolim de Moura", "location": "Brazil", "station_no": 73, "lat": -11.5820, "lon": -61.7730, "elevation": 252.0, "established": "2007-01-01", "closed": "2007-12-31", "surface_type": "concrete", "topography_type": "flat, rural", "scientist": "Enio Bueno Pereira", "email": "enio.pereira@inpe.br", "url": None},
    "RUN": {"name": "Reunion Island, University", "location": "Reunion", "station_no": None, "lat": -20.9014, "lon": 55.4836, "elevation": 116.0, "established": None, "closed": None, "surface_type": None, "topography_type": None, "scientist": "Béatrice Morel", "email": "beatrice.morel@univ-reunion.fr", "url": None},
    "SAP": {"name": "Sapporo", "location": "Japan", "station_no": 4, "lat": 43.0600, "lon": 141.3286, "elevation": 17.2, "established": "2010-04-01", "closed": None, "surface_type": "asphalt", "topography_type": "flat, urban", "scientist": "Kouichi Nakashima", "email": "rrc-jma@met.kishou.go.jp", "url": "http://hdl.handle.net/10013/epic.44630.d028"},
    "SBO": {"name": "Sede Boqer", "location": "Israel", "station_no": 43, "lat": 30.8597, "lon": 34.7794, "elevation": 500.0, "established": "2003-01-01", "closed": "2012-12-31", "surface_type": "desert rock", "topography_type": "hilly, rural", "scientist": "Dr. Nurit Agam", "email": "agam@bgu.ac.il", "url": None},
    "SEL": {"name": "Selegua", "location": "Mexico", "station_no": 83, "lat": 15.7840, "lon": -91.9902, "elevation": 602.0, "established": None, "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Roberto Bonifaz", "email": "bonifaz@unam.mx", "url": None},
    "SMS": {"name": "São Martinho da Serra", "location": "Brazil", "station_no": 70, "lat": -29.4428, "lon": -53.8231, "elevation": 489.0, "established": "2006-01-01", "closed": None, "surface_type": "concrete, since 2015: grass", "topography_type": "flat, rural", "scientist": "Enio Bueno Pereira", "email": "enio.pereira@inpe.br", "url": None},
    "SON": {"name": "Sonnblick", "location": "Austria", "station_no": 75, "lat": 47.0540, "lon": 12.9577, "elevation": 3108.9, "established": "2013-01-01", "closed": None, "surface_type": "rock", "topography_type": "mountain top, rural", "scientist": "Marc Olefs", "email": "marc.olefs@zamg.ac.at", "url": "http://www.sonnblick.net"},
    "SOV": {"name": "Solar Village", "location": "Saudi Arabia", "station_no": 41, "lat": 24.9100, "lon": 46.4100, "elevation": 650.0, "established": "1998-08-01", "closed": None, "surface_type": "desert, sand", "topography_type": "flat, rural", "scientist": None, "email": None, "url": None},
    "SPO": {"name": "South Pole", "location": "Antarctica", "station_no": 26, "lat": -89.9830, "lon": -24.7990, "elevation": 2800.0, "established": "1992-01-01", "closed": None, "surface_type": "glacier, accumulation area", "topography_type": "flat, rural", "scientist": "Laura Riihimaki", "email": "laura.riihimaki@noaa.gov", "url": "http://www.esrl.noaa.gov/gmd/obop/spo/"},
    "SXF": {"name": "Sioux Falls", "location": "South Dakota, United States of America", "station_no": 37, "lat": 43.7300, "lon": -96.6200, "elevation": 473.0, "established": "2003-06-01", "closed": None, "surface_type": "grass", "topography_type": "hilly, rural", "scientist": "John A. Augustine", "email": "john.a.augustine@noaa.gov", "url": "http://www.srrb.noaa.gov/surfrad/siouxfalls.html"},
    "SYO": {"name": "Syowa", "location": "Cosmonaut Sea", "station_no": 17, "lat": -69.0050, "lon": 39.5890, "elevation": 18.0, "established": "1994-01-01", "closed": None, "surface_type": "sea ice", "topography_type": "hilly, rural", "scientist": "Yoshinobu Tanaka", "email": "antarctic@met.kishou.go.jp", "url": "http://www.nipr.ac.jp/english/"},
    "TAM": {"name": "Tamanrasset", "location": "Algeria", "station_no": 42, "lat": 22.7903, "lon": 5.5292, "elevation": 1385.0, "established": "2000-03-01", "closed": None, "surface_type": "desert, rock", "topography_type": "flat, rural", "scientist": "Sidi BAIKA", "email": "s.baika@meteo.dz", "url": None},
    "TAT": {"name": "Tateno", "location": "Japan", "station_no": 16, "lat": 36.0581, "lon": 140.1258, "elevation": 25.0, "established": "1996-02-01", "closed": None, "surface_type": "grass", "topography_type": "flat, urban", "scientist": "Osamu Ijima", "email": "ijima@met.kishou.go.jp", "url": "http://hdl.handle.net/10013/epic.44630.d027"},
    "TIK": {"name": "Tiksi", "location": "Siberia, Russia", "station_no": 48, "lat": 71.5862, "lon": 128.9188, "elevation": 48.0, "established": "2010-06-08", "closed": None, "surface_type": "tundra", "topography_type": "flat, rural", "scientist": "Vasilii Kustov", "email": "kustov@aari.ru", "url": "http://www.aari.ru/main.php?lg=1"},
    "TIR": {"name": "Tiruvallur", "location": "India", "station_no": 59, "lat": 13.0923, "lon": 79.9738, "elevation": 36.0, "established": "2014-04-16", "closed": None, "surface_type": "rock", "topography_type": "urban", "scientist": "Prasun Kumar", "email": "prasun.niwe@nic.in", "url": None},
    "TOR": {"name": "Toravere", "location": "Estonia", "station_no": 9, "lat": 58.2540, "lon": 26.4620, "elevation": 70.0, "established": "1999-01-01", "closed": None, "surface_type": "grass", "topography_type": "flat, rural", "scientist": "Ain Kallis", "email": "kallis@aai.ee", "url": "http://www.aai.ee"},
    "XIA": {"name": "Xianghe", "location": "China", "station_no": 44, "lat": 39.7540, "lon": 116.9620, "elevation": 32.0, "established": "2005-01-01", "closed": "2016-12-31", "surface_type": "desert, rock", "topography_type": "flat, rural", "scientist": "Xiangao Xia", "email": "xxa@mail.iap.ac.cn", "url": None},
    "YUS": {"name": "Yushan", "location": "Taiwan", "station_no": 84, "lat": 23.4876, "lon": 120.9595, "elevation": 3858.0, "established": None, "closed": None, "surface_type": "forest, mixed", "topography_type": "mountain top, rural", "scientist": "Kun-Wei Lin", "email": "adenins@cwb.gov.tw", "url": None}
}

# QUANTITIES: measured quantity label → WRMC ``radiationQuantityMeasured`` id for LR0008. R A3.
# 辐射/气象量名称 → LR0008 中 ``radiationQuantityMeasured`` 的编号（R A3）。
QUANTITIES = {
    "global (pyranometer)": 2,
    "direct (pyheliometer)": 3,
    "diffuse (pyranometer)": 4,
    "long-wave downward": 5,
    "air temperature": 21,
    "relative humidity": 22,
    "pressure": 23,
    "uv-a-global": 121,
    "uv-b-direct": 122,
    "uv-b-global": 123,
    "uv-b-diffuse": 124,
    "uv-b-reflected": 125,
    "short-wave reflected": 131,
    "long-wave upward": 132,
    "net radiation (net radiometer)": 141,
    "global (pyranometer) secondary": 2000700,
    "short-wave reflected secondary": 131000700,
    "long-wave upward secondary": 132000700,
    "long-wave downward secondary": 5000700,
    "air temperature secondary": 21000700,
    "relative humidity secondary": 22000700,
    "short-wave reflected tertiary": 131003000,
    "short-wave spec. bd. 1": 104,
    "short-wave spec. bd. 3": 112,
    "total cloud amount with instrument": 301,
    "cloud base height with instrument": 302,
    "cloud liquid water": 303
}

# SURFACES: surface-type label → I2 code for LR0004 ``surfaceType``. R A4.
# 地表类型描述 → LR0004 中 ``surfaceType`` 的整数码（R A4）。
SURFACES = {
    "glacier - accumulation area": 1,
    "glacier - ablation area": 2,
    "iceshelf": 3,
    "sea ice": 4,
    "water - river": 5,
    "water - lake": 6,
    "water - ocean": 7,
    "desert - rock": 8,
    "desert - sand": 9,
    "desert - gravel": 10,
    "concrete": 11,
    "asphalt": 12,
    "cultivated": 13,
    "tundra": 14,
    "grass": 15,
    "shrub": 16,
    "forest - evergreen": 17,
    "forest - deciduous": 18,
    "forest - mixed": 19,
    "rock": 20,
    "sand": 21
}


# TOPOGRAPHIES: human-readable topography label → I2 code for LR0004 ``topographyType``. R A5.
# 地形描述字符串 → LR0004 中 ``topographyType`` 的整数码（R A5）。
TOPOGRAPHIES = {
    "flat - urban": 1,
    "flat - rural": 2,
    "hilly - urban": 3,
    "hilly - rural": 4,
    "mountain top - urban": 5,
    "mountain top - rural": 6,
    "mountain valley - urban": 7,
    "mountain valley - rural": 8
}

# PYRGEOMETER_BODY: pyrgeometer body compensation option → I2 code for LR0008 ``pyrgeometerBody``. R A6.
# 长波表体补偿方式 → LR0008 ``pyrgeometerBody`` 编码（R A6）。
PYRGEOMETER_BODY = {
    "Manufacturer & battery circuit": 1,
    "Corrected manufacturer & battery circuit": 2,
    "Temperature measurement with sigma Tc^4": 3,
    "Other": 4
}

# PYRGEOMETER_DOME: pyrgeometer dome / ventilation option → I2 code for LR0008 ``pyrgeometerDome``. R A7.
# 长波表罩与通风组合 → LR0008 ``pyrgeometerDome`` 编码（R A7）。
PYRGEOMETER_DOME = {
    "Dome shaded": 1,
    "Instrument ventilated": 2,
    "Temperature measurement with sigma Tc^4": 3,
    "Shaded & ventilated": 4,
    "Shaded & sigma Tc^4": 5,
    "Ventilated & sigma Tc^4": 6,
    "Shaded & ventilated & sigma Tc^4": 7,
    "Other": 8
}
