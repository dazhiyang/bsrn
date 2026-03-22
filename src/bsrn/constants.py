"""
BSRN Standards and Constants.
BSRN 标准和常量。

Contains physical constants, model parameters, and station metadata.
包含物理常量、模型参数和站点元数据。
"""

"""
BSRN Data Services: Connectivity parameters for the BSRN archive.
BSRN 数据服务：BSRN 存档的连接参数。
"""
BSRN_FTP_HOST = "ftp.bsrn.awi.de"

"""
Station Metadata: Mapping of station abbreviations to geographic info.
站点元数据：站点缩写到地理信息的映射。

Includes: name, latitude [degrees], longitude [degrees], elevation [m].
包括：名称、纬度 [度]、经度 [度]、海拔 [米]。

References
----------
Driemel, A., et al. (2018). Baseline Surface Radiation Network (BSRN): 
structure and data description (1992–2017). Earth System Science Data, 
10(3), 1491-1501.
"""
BSRN_STATIONS = {
    "ABS": {"name": "Abashiri", "lat": 44.0178, "lon": 144.2797, "elev": 38.0, "status": "active", "kgc": "Dfb"},
    "ALE": {"name": "Alert", "lat": 82.49, "lon": -62.42, "elev": 127.0, "status": "closed", "kgc": "ET"},
    "ASP": {"name": "Alice Springs", "lat": -23.798, "lon": 133.888, "elev": 547.0, "status": "inactive", "kgc": "BWh"},
    "BAR": {"name": "Barrow", "lat": 71.323, "lon": -156.607, "elev": 8.0, "status": "active", "kgc": "ET"},
    "BER": {"name": "Bermuda", "lat": 32.3008, "lon": -64.766, "elev": 8.0, "status": "active", "kgc": "Af"},
    "BIL": {"name": "Billings", "lat": 36.605, "lon": -97.516, "elev": 317.0, "status": "active", "kgc": "Cfa"},
    "BON": {"name": "Bondville", "lat": 40.0667, "lon": -88.3667, "elev": 213.0, "status": "active", "kgc": "Dfa"},
    "BOS": {"name": "Boulder (BOS)", "lat": 40.125, "lon": -105.237, "elev": 1689.0, "status": "active", "kgc": "BSk"},
    "BOU": {"name": "Boulder (BOU)", "lat": 40.05, "lon": -105.007, "elev": 1577.0, "status": "closed", "kgc": "BSk"},
    "BRB": {"name": "Brasilia", "lat": -15.601, "lon": -47.713, "elev": 1023.0, "status": "inactive", "kgc": "Aw"},
    "BUD": {"name": "Budapest-Lorinc", "lat": 47.4291, "lon": 19.1822, "elev": 139.1, "status": "active", "kgc": "Dfb"},
    "CAB": {"name": "Cabauw", "lat": 51.968, "lon": 4.928, "elev": 0.0, "status": "active", "kgc": "Cfb"},
    "CAM": {"name": "Camborne", "lat": 50.2167, "lon": -5.3167, "elev": 88.0, "status": "inactive", "kgc": "Cfb"},
    "CAP": {"name": "Cape Baranova", "lat": 79.27, "lon": 101.75, "elev": 25.0, "status": "closed", "kgc": "ET"},
    "CAR": {"name": "Carpentras", "lat": 44.083, "lon": 5.059, "elev": 100.0, "status": "closed", "kgc": "Csa"},
    "CLH": {"name": "Chesapeake Light", "lat": 36.905, "lon": -75.713, "elev": 37.0, "status": "closed", "kgc": ""},
    "CNR": {"name": "Cener", "lat": 42.816, "lon": -1.601, "elev": 471.0, "status": "active", "kgc": "Cfb"},
    "COC": {"name": "Cocos Island", "lat": -12.193, "lon": 96.835, "elev": 6.0, "status": "inactive", "kgc": "Af"},
    "DAA": {"name": "De Aar", "lat": -30.6667, "lon": 23.993, "elev": 1287.0, "status": "inactive", "kgc": "BSk"},
    "DAR": {"name": "Darwin", "lat": -12.425, "lon": 130.891, "elev": 30.0, "status": "closed", "kgc": "Aw"},
    "DOM": {"name": "Concordia Station, Dome C", "lat": -75.1, "lon": 123.383, "elev": 3233.0, "status": "active", "kgc": "EF"},
    "DRA": {"name": "Desert Rock", "lat": 36.626, "lon": -116.018, "elev": 1007.0, "status": "active", "kgc": "BWk"},
    "DWN": {"name": "Darwin Met Office", "lat": -12.424, "lon": 130.8925, "elev": 32.0, "status": "inactive", "kgc": "Aw"},
    "E13": {"name": "Southern Great Plains", "lat": 36.605, "lon": -97.485, "elev": 318.0, "status": "active", "kgc": "Cfa"},
    "ENA": {"name": "Eastern North Atlantic", "lat": 39.0911, "lon": -28.0292, "elev": 15.2, "status": "inactive", "kgc": "Csa"},
    "EUR": {"name": "Eureka", "lat": 79.989, "lon": -85.9404, "elev": 85.0, "status": "closed", "kgc": "ET"},
    "FLO": {"name": "Florianopolis", "lat": -27.6047, "lon": -48.5227, "elev": 11.0, "status": "active", "kgc": "Cfa"},
    "FPE": {"name": "Fort Peck", "lat": 48.3167, "lon": -105.1, "elev": 634.0, "status": "active", "kgc": "BSk"},
    "FUA": {"name": "Fukuoka", "lat": 33.5822, "lon": 130.3764, "elev": 3.0, "status": "closed", "kgc": "Cfa"},
    "GAN": {"name": "Gandhinagar", "lat": 23.1101, "lon": 72.6276, "elev": 65.0, "status": "closed", "kgc": "BSh"},
    "GCR": {"name": "Goodwin Creek", "lat": 34.2547, "lon": -89.8729, "elev": 98.0, "status": "active", "kgc": "Cfa"},
    "GIM": {"name": "Granite Island", "lat": 46.721, "lon": -87.411, "elev": 208.0, "status": "active", "kgc": "Dfb"},
    "GOB": {"name": "Gobabeb", "lat": -23.5614, "lon": 15.042, "elev": 407.0, "status": "active", "kgc": "BWh"},
    "GUR": {"name": "Gurgaon", "lat": 28.4249, "lon": 77.156, "elev": 259.0, "status": "closed", "kgc": "BSh"},
    "GVN": {"name": "Georg von Neumayer", "lat": -70.65, "lon": -8.25, "elev": 42.0, "status": "active", "kgc": ""},
    "HOW": {"name": "Howrah", "lat": 22.5535, "lon": 88.3064, "elev": 51.0, "status": "closed", "kgc": "Aw"},
    "ILO": {"name": "Ilorin", "lat": 8.5333, "lon": 4.5667, "elev": 350.0, "status": "closed", "kgc": "Aw"},
    "INO": {"name": "Marguele", "lat": 44.3439, "lon": 26.0123, "elev": 110.0, "status": "active", "kgc": "Dfa"},
    "ISH": {"name": "Ishigakijima", "lat": 24.3367, "lon": 124.1644, "elev": 5.7, "status": "active", "kgc": "Af"},
    "IZA": {"name": "Izaña", "lat": 28.3093, "lon": -16.4993, "elev": 2372.9, "status": "active", "kgc": "Csb"},
    "KWA": {"name": "Kwajalein", "lat": 8.72, "lon": 167.731, "elev": 10.0, "status": "closed", "kgc": "Af"},
    "LAU": {"name": "Lauder", "lat": -45.045, "lon": 169.689, "elev": 350.0, "status": "inactive", "kgc": "Cfb"},
    "LER": {"name": "Lerwick", "lat": 60.1389, "lon": -1.1847, "elev": 80.0, "status": "inactive", "kgc": "Cfc"},
    "LIN": {"name": "Lindenberg", "lat": 52.21, "lon": 14.122, "elev": 125.0, "status": "active", "kgc": "Dfb"},
    "LMP": {"name": "Lampedusa", "lat": 35.518, "lon": 12.63, "elev": 50.0, "status": "active", "kgc": "BSh"},
    "LRC": {"name": "Langley Research Center", "lat": 37.1038, "lon": -76.3872, "elev": 3.0, "status": "active", "kgc": "Cfa"},
    "LYU": {"name": "Lanyu Station", "lat": 22.037, "lon": 121.5583, "elev": 324.0, "status": "active", "kgc": "Af"},
    "MAN": {"name": "Momote", "lat": -2.058, "lon": 147.425, "elev": 6.0, "status": "closed", "kgc": "Af"},
    "MNM": {"name": "Minamitorishima", "lat": 24.2883, "lon": 153.9833, "elev": 7.1, "status": "active", "kgc": ""},
    "NAU": {"name": "Nauru Island", "lat": -0.521, "lon": 166.9167, "elev": 7.0, "status": "closed", "kgc": "Af"},
    "NEW": {"name": "Newcastle", "lat": -32.8842, "lon": 151.7289, "elev": 18.5, "status": "inactive", "kgc": "Cfa"},
    "NYA": {"name": "Ny-Ålesund", "lat": 78.9227, "lon": 11.9273, "elev": 11.0, "status": "active", "kgc": "ET"},
    "OHY": {"name": "Observatory of Huancayo", "lat": -12.05, "lon": -75.32, "elev": 3314.0, "status": "active", "kgc": "Cwb"},
    "PAL": {"name": "Palaiseau, SIRTA Observatory", "lat": 48.713, "lon": 2.208, "elev": 156.0, "status": "active", "kgc": "Cfb"},
    "PAR": {"name": "Paramaribo", "lat": 5.806, "lon": -55.2146, "elev": 4.0, "status": "active", "kgc": "Af"},
    "PAY": {"name": "Payerne", "lat": 46.8123, "lon": 6.9422, "elev": 491.0, "status": "active", "kgc": "Dfb"},
    "PSU": {"name": "Rock Springs", "lat": 40.72, "lon": -77.9333, "elev": 376.0, "status": "active", "kgc": "Dfa"},
    "PTR": {"name": "Petrolina", "lat": -9.069, "lon": -40.32, "elev": 387.0, "status": "inactive", "kgc": "BSh"},
    "QIQ": {"name": "Qiqihar", "lat": 47.7957, "lon": 124.4852, "elev": 170.0, "status": "active", "kgc": "Dwa"},
    "REG": {"name": "Regina", "lat": 50.205, "lon": -104.713, "elev": 578.0, "status": "closed", "kgc": "Dfb"},
    "RLM": {"name": "Rolim de Moura", "lat": -11.582, "lon": -61.773, "elev": 252.0, "status": "closed", "kgc": "Aw"},
    "RUN": {"name": "Reunion Island, University", "lat": -20.9014, "lon": 55.4836, "elev": 116.0, "status": "active", "kgc": "Am"},
    "SAP": {"name": "Sapporo", "lat": 43.06, "lon": 141.3286, "elev": 17.2, "status": "closed", "kgc": "Dfb"},
    "SBO": {"name": "Sede Boqer", "lat": 30.8597, "lon": 34.7794, "elev": 500.0, "status": "closed", "kgc": "BWh"},
    "SEL": {"name": "Selegua", "lat": 15.784, "lon": -91.9902, "elev": 602.0, "status": "active", "kgc": "Aw"},
    "SMS": {"name": "São Martinho da Serra", "lat": -29.4428, "lon": -53.8231, "elev": 489.0, "status": "inactive", "kgc": "Cfa"},
    "SON": {"name": "Sonnblick", "lat": 47.054, "lon": 12.9577, "elev": 3108.9, "status": "active", "kgc": "ET"},
    "SOV": {"name": "Solar Village", "lat": 24.91, "lon": 46.41, "elev": 650.0, "status": "closed", "kgc": "BWh"},
    "SPO": {"name": "South Pole", "lat": -89.983, "lon": -24.799, "elev": 2800.0, "status": "active", "kgc": "EF"},
    "SXF": {"name": "Sioux Falls", "lat": 43.73, "lon": -96.62, "elev": 473.0, "status": "active", "kgc": "Dfa"},
    "SYO": {"name": "Syowa", "lat": -69.0053, "lon": 39.5811, "elev": 29.0, "status": "active", "kgc": "EF"},
    "TAM": {"name": "Tamanrasset", "lat": 22.7903, "lon": 5.5292, "elev": 1385.0, "status": "active", "kgc": "BWh"},
    "TAT": {"name": "Tateno", "lat": 36.0581, "lon": 140.1258, "elev": 25.0, "status": "active", "kgc": "Cfa"},
    "TIK": {"name": "Tiksi", "lat": 71.5862, "lon": 128.9188, "elev": 48.0, "status": "closed", "kgc": "ET"},
    "TIR": {"name": "Tiruvallur", "lat": 13.0923, "lon": 79.9738, "elev": 36.0, "status": "closed", "kgc": "Aw"},
    "TNB": {"name": "Terra Nova Bay", "lat": -74.6223, "lon": 164.2283, "elev": 28.0, "status": "candidate", "kgc": "Aw"},
    "TOR": {"name": "Toravere", "lat": 58.2641, "lon": 26.4613, "elev": 70.0, "status": "active", "kgc": "Dfb"},
    "XIA": {"name": "Xianghe", "lat": 39.754, "lon": 116.962, "elev": 32.0, "status": "closed", "kgc": "Dwa"},
    "YUS": {"name": "Yushan Station", "lat": 23.4876, "lon": 120.9595, "elev": 3858.0, "status": "active", "kgc": "ET"},
}

"""
Linke Turbidity: Monthly values for BSRN stations.
Linke 浑浊度：BSRN 站点的月度值。

References
----------
SoDA-PRO (2024). Linke Turbidity Factor. Retrieved from 
https://www.soda-pro.com/help/general-knowledge/linke-turbidity-factor
"""
LINKE_TURBIDITY = {
    "ABS": {"Jan": 3.6, "Feb": 3.6, "Mar": 4.4, "Apr": 5.4, "May": 5.0, "Jun": 4.1, "Jul": 3.9, "Aug": 3.6, "Sep": 3.1, "Oct": 3.2, "Nov": 3.2, "Dec": 3.2},
    "ALE": {"Jan": 1.0, "Feb": 1.0, "Mar": 3.2, "Apr": 3.2, "May": 3.0, "Jun": 3.2, "Jul": 2.7, "Aug": 2.8, "Sep": 3.0, "Oct": 1.0, "Nov": 1.0, "Dec": 1.0},
    "ASP": {"Jan": 3.2, "Feb": 3.1, "Mar": 2.9, "Apr": 2.9, "May": 2.8, "Jun": 2.6, "Jul": 2.6, "Aug": 2.6, "Sep": 3.0, "Oct": 3.0, "Nov": 3.3, "Dec": 3.3},
    "BAR": {"Jan": 1.1, "Feb": 3.5, "Mar": 3.3, "Apr": 2.9, "May": 3.5, "Jun": 2.9, "Jul": 2.9, "Aug": 3.0, "Sep": 2.9, "Oct": 2.8, "Nov": 1.1, "Dec": 1.1},
    "BER": {"Jan": 3.4, "Feb": 3.4, "Mar": 3.4, "Apr": 3.7, "May": 3.9, "Jun": 3.7, "Jul": 4.1, "Aug": 3.7, "Sep": 3.6, "Oct": 3.5, "Nov": 3.4, "Dec": 3.4},
    "BIL": {"Jan": 2.6, "Feb": 2.8, "Mar": 2.9, "Apr": 3.4, "May": 3.6, "Jun": 3.6, "Jul": 3.8, "Aug": 3.8, "Sep": 3.4, "Oct": 2.8, "Nov": 2.7, "Dec": 2.5},
    "BON": {"Jan": 2.8, "Feb": 3.1, "Mar": 3.2, "Apr": 3.6, "May": 4.3, "Jun": 3.7, "Jul": 4.4, "Aug": 4.3, "Sep": 3.6, "Oct": 3.1, "Nov": 2.9, "Dec": 2.7},
    "BOS": {"Jan": 3.1, "Feb": 3.3, "Mar": 3.5, "Apr": 3.9, "May": 3.8, "Jun": 3.9, "Jul": 3.9, "Aug": 3.6, "Sep": 3.2, "Oct": 3.0, "Nov": 2.9, "Dec": 3.0},
    "BOU": {"Jan": 3.1, "Feb": 3.3, "Mar": 3.5, "Apr": 3.9, "May": 3.8, "Jun": 3.9, "Jul": 3.9, "Aug": 3.6, "Sep": 3.2, "Oct": 3.0, "Nov": 2.9, "Dec": 3.0},
    "BRB": {"Jan": 3.0, "Feb": 3.2, "Mar": 3.3, "Apr": 2.9, "May": 2.9, "Jun": 3.4, "Jul": 3.4, "Aug": 4.2, "Sep": 5.1, "Oct": 5.3, "Nov": 3.7, "Dec": 3.9},
    "BUD": {"Jan": 2.7, "Feb": 2.9, "Mar": 3.3, "Apr": 3.5, "May": 3.6, "Jun": 3.8, "Jul": 4.0, "Aug": 3.7, "Sep": 3.5, "Oct": 3.3, "Nov": 3.0, "Dec": 2.8},
    "CAB": {"Jan": 3.0, "Feb": 3.2, "Mar": 3.9, "Apr": 3.9, "May": 4.1, "Jun": 4.0, "Jul": 3.7, "Aug": 4.2, "Sep": 3.1, "Oct": 3.3, "Nov": 3.3, "Dec": 2.4},
    "CAM": {"Jan": 3.4, "Feb": 3.3, "Mar": 3.9, "Apr": 3.7, "May": 3.9, "Jun": 3.4, "Jul": 3.4, "Aug": 3.6, "Sep": 3.7, "Oct": 3.6, "Nov": 3.4, "Dec": 3.4},
    "CAP": {"Jan": 1.0, "Feb": 1.0, "Mar": 3.5, "Apr": 2.9, "May": 3.7, "Jun": 3.1, "Jul": 2.7, "Aug": 3.4, "Sep": 3.2, "Oct": 1.9, "Nov": 1.0, "Dec": 1.0},
    "CAR": {"Jan": 2.9, "Feb": 3.0, "Mar": 3.3, "Apr": 3.3, "May": 3.7, "Jun": 3.6, "Jul": 3.5, "Aug": 3.6, "Sep": 3.4, "Oct": 3.0, "Nov": 2.9, "Dec": 2.9},
    "CLH": {"Jan": 2.6, "Feb": 2.7, "Mar": 2.9, "Apr": 3.4, "May": 3.9, "Jun": 4.2, "Jul": 4.5, "Aug": 4.6, "Sep": 3.3, "Oct": 2.9, "Nov": 2.7, "Dec": 2.5},
    "CNR": {"Jan": 2.8, "Feb": 3.1, "Mar": 3.5, "Apr": 3.4, "May": 3.6, "Jun": 3.9, "Jul": 3.7, "Aug": 3.7, "Sep": 3.3, "Oct": 3.2, "Nov": 2.9, "Dec": 2.8},
    "COC": {"Jan": 4.1, "Feb": 3.8, "Mar": 3.8, "Apr": 4.4, "May": 4.2, "Jun": 4.2, "Jul": 4.4, "Aug": 4.4, "Sep": 4.6, "Oct": 4.7, "Nov": 4.8, "Dec": 4.1},
    "DAA": {"Jan": 2.9, "Feb": 2.9, "Mar": 2.8, "Apr": 2.8, "May": 2.7, "Jun": 2.6, "Jul": 2.6, "Aug": 2.8, "Sep": 2.9, "Oct": 2.9, "Nov": 2.9, "Dec": 2.8},
    "DAR": {"Jan": 5.6, "Feb": 5.7, "Mar": 5.1, "Apr": 3.3, "May": 3.1, "Jun": 3.0, "Jul": 3.0, "Aug": 3.2, "Sep": 3.6, "Oct": 3.9, "Nov": 4.3, "Dec": 5.4},
    "DOM": {"Jan": 2.1, "Feb": 2.7, "Mar": 2.7, "Apr": 1.1, "May": 1.1, "Jun": 1.1, "Jul": 1.1, "Aug": 1.1, "Sep": 2.7, "Oct": 2.7, "Nov": 2.7, "Dec": 0.9},
    "DRA": {"Jan": 3.8, "Feb": 4.0, "Mar": 4.9, "Apr": 5.6, "May": 5.7, "Jun": 5.0, "Jul": 5.1, "Aug": 4.8, "Sep": 4.3, "Oct": 4.0, "Nov": 3.5, "Dec": 3.3},
    "DWN": {"Jan": 5.6, "Feb": 5.7, "Mar": 5.1, "Apr": 3.3, "May": 3.1, "Jun": 3.0, "Jul": 3.0, "Aug": 3.2, "Sep": 3.6, "Oct": 3.9, "Nov": 4.3, "Dec": 5.4},
    "E13": {"Jan": 2.6, "Feb": 2.8, "Mar": 2.9, "Apr": 3.4, "May": 3.6, "Jun": 3.6, "Jul": 3.8, "Aug": 3.8, "Sep": 3.4, "Oct": 2.8, "Nov": 2.7, "Dec": 2.5},
    "ENA": {"Jan": 3.8, "Feb": 3.9, "Mar": 3.8, "Apr": 4.1, "May": 3.9, "Jun": 3.6, "Jul": 3.5, "Aug": 3.4, "Sep": 3.3, "Oct": 3.7, "Nov": 3.6, "Dec": 3.4},
    "EUR": {"Jan": 1.0, "Feb": 1.0, "Mar": 3.0, "Apr": 3.0, "May": 3.0, "Jun": 3.5, "Jul": 3.2, "Aug": 3.2, "Sep": 2.5, "Oct": 1.0, "Nov": 1.0, "Dec": 1.0},
    "FLO": {"Jan": 3.3, "Feb": 3.2, "Mar": 3.1, "Apr": 3.0, "May": 2.8, "Jun": 2.9, "Jul": 2.9, "Aug": 3.1, "Sep": 3.5, "Oct": 3.3, "Nov": 3.3, "Dec": 3.4},
    "FPE": {"Jan": 3.0, "Feb": 3.3, "Mar": 3.5, "Apr": 4.1, "May": 4.2, "Jun": 3.3, "Jul": 3.3, "Aug": 3.9, "Sep": 3.5, "Oct": 2.9, "Nov": 2.7, "Dec": 3.2},
    "FUA": {"Jan": 3.9, "Feb": 4.7, "Mar": 5.0, "Apr": 5.6, "May": 5.4, "Jun": 6.2, "Jul": 5.0, "Aug": 4.4, "Sep": 3.8, "Oct": 3.1, "Nov": 3.1, "Dec": 3.6},
    "GAN": {"Jan": 4.5, "Feb": 4.7, "Mar": 5.2, "Apr": 5.8, "May": 6.7, "Jun": 7.1, "Jul": 8.7, "Aug": 6.6, "Sep": 6.1, "Oct": 4.9, "Nov": 4.8, "Dec": 4.3},
    "GCR": {"Jan": 2.7, "Feb": 2.9, "Mar": 3.1, "Apr": 3.4, "May": 3.6, "Jun": 3.8, "Jul": 4.3, "Aug": 4.3, "Sep": 3.4, "Oct": 3.2, "Nov": 2.7, "Dec": 2.7},
    "GIM": {"Jan": 3.2, "Feb": 3.1, "Mar": 3.3, "Apr": 3.5, "May": 3.8, "Jun": 3.5, "Jul": 3.5, "Aug": 3.3, "Sep": 3.7, "Oct": 3.1, "Nov": 3.0, "Dec": 2.6},
    "GOB": {"Jan": 3.9, "Feb": 4.2, "Mar": 3.9, "Apr": 3.9, "May": 4.0, "Jun": 4.2, "Jul": 5.3, "Aug": 3.7, "Sep": 3.9, "Oct": 4.0, "Nov": 3.9, "Dec": 3.5},
    "GUR": {"Jan": 5.8, "Feb": 5.4, "Mar": 5.7, "Apr": 7.6, "May": 9.8, "Jun": 9.8, "Jul": 9.8, "Aug": 9.0, "Sep": 6.5, "Oct": 7.4, "Nov": 6.5, "Dec": 5.9},
    "GVN": {"Jan": 2.6, "Feb": 2.7, "Mar": 3.1, "Apr": 2.1, "May": 1.1, "Jun": 1.1, "Jul": 1.1, "Aug": 1.1, "Sep": 3.8, "Oct": 3.6, "Nov": 3.1, "Dec": 2.8},
    "HOW": {"Jan": 7.5, "Feb": 7.2, "Mar": 6.7, "Apr": 7.8, "May": 9.3, "Jun": 8.9, "Jul": 7.3, "Aug": 7.6, "Sep": 6.5, "Oct": 5.5, "Nov": 6.4, "Dec": 7.0},
    "ILO": {"Jan": 7.8, "Feb": 8.7, "Mar": 8.5, "Apr": 8.2, "May": 6.9, "Jun": 6.9, "Jul": 5.8, "Aug": 5.0, "Sep": 5.5, "Oct": 6.4, "Nov": 7.0, "Dec": 7.1},
    "INO": {"Jan": 2.8, "Feb": 3.4, "Mar": 3.2, "Apr": 4.1, "May": 3.8, "Jun": 3.6, "Jul": 3.9, "Aug": 4.4, "Sep": 3.9, "Oct": 3.3, "Nov": 3.5, "Dec": 2.9},
    "ISH": {"Jan": 4.3, "Feb": 4.3, "Mar": 5.2, "Apr": 5.4, "May": 4.8, "Jun": 4.2, "Jul": 3.9, "Aug": 4.0, "Sep": 4.3, "Oct": 4.2, "Nov": 4.2, "Dec": 4.1},
    "IZA": {"Jan": 2.8, "Feb": 2.9, "Mar": 2.9, "Apr": 3.1, "May": 3.0, "Jun": 3.1, "Jul": 4.4, "Aug": 4.0, "Sep": 3.6, "Oct": 3.0, "Nov": 3.1, "Dec": 2.7},
    "KWA": {"Jan": 4.4, "Feb": 4.3, "Mar": 4.4, "Apr": 4.5, "May": 4.3, "Jun": 4.2, "Jul": 3.7, "Aug": 3.7, "Sep": 3.7, "Oct": 3.6, "Nov": 3.9, "Dec": 4.3},
    "LAU": {"Jan": 3.4, "Feb": 3.1, "Mar": 3.0, "Apr": 2.8, "May": 2.7, "Jun": 2.8, "Jul": 2.8, "Aug": 2.9, "Sep": 3.0, "Oct": 3.1, "Nov": 3.3, "Dec": 3.4},
    "LER": {"Jan": 3.0, "Feb": 3.4, "Mar": 3.7, "Apr": 4.1, "May": 3.7, "Jun": 3.4, "Jul": 3.4, "Aug": 3.7, "Sep": 3.7, "Oct": 3.4, "Nov": 3.5, "Dec": 1.7},
    "LIN": {"Jan": 2.9, "Feb": 2.7, "Mar": 3.5, "Apr": 3.7, "May": 3.4, "Jun": 3.3, "Jul": 3.4, "Aug": 3.8, "Sep": 3.3, "Oct": 3.5, "Nov": 3.4, "Dec": 2.9},
    "LMP": {"Jan": 3.2, "Feb": 3.7, "Mar": 5.9, "Apr": 3.6, "May": 4.3, "Jun": 4.5, "Jul": 5.9, "Aug": 4.6, "Sep": 5.1, "Oct": 5.1, "Nov": 3.6, "Dec": 3.3},
    "LRC": {"Jan": 2.4, "Feb": 2.6, "Mar": 2.8, "Apr": 3.4, "May": 3.8, "Jun": 4.2, "Jul": 4.5, "Aug": 4.6, "Sep": 3.2, "Oct": 2.7, "Nov": 2.5, "Dec": 2.4},
    "LYU": {"Jan": 4.3, "Feb": 4.4, "Mar": 5.6, "Apr": 5.3, "May": 4.5, "Jun": 4.0, "Jul": 3.9, "Aug": 4.0, "Sep": 4.5, "Oct": 4.5, "Nov": 4.3, "Dec": 4.3},
    "MAN": {"Jan": 3.7, "Feb": 3.7, "Mar": 3.7, "Apr": 3.5, "May": 3.4, "Jun": 3.6, "Jul": 3.7, "Aug": 3.8, "Sep": 3.7, "Oct": 3.8, "Nov": 3.5, "Dec": 3.6},
    "MNM": {"Jan": 4.1, "Feb": 3.9, "Mar": 4.2, "Apr": 4.5, "May": 4.1, "Jun": 3.7, "Jul": 3.8, "Aug": 3.9, "Sep": 3.7, "Oct": 3.7, "Nov": 3.9, "Dec": 4.0},
    "NAU": {"Jan": 3.9, "Feb": 3.9, "Mar": 4.0, "Apr": 3.6, "May": 3.6, "Jun": 3.4, "Jul": 3.7, "Aug": 3.7, "Sep": 3.4, "Oct": 3.4, "Nov": 3.5, "Dec": 3.7},
    "NEW": {"Jan": 3.7, "Feb": 3.4, "Mar": 3.1, "Apr": 2.9, "May": 2.6, "Jun": 2.6, "Jul": 2.5, "Aug": 2.7, "Sep": 2.8, "Oct": 3.0, "Nov": 3.2, "Dec": 3.3},
    "NYA": {"Jan": 1.1, "Feb": 1.1, "Mar": 3.3, "Apr": 3.5, "May": 3.3, "Jun": 2.9, "Jul": 2.9, "Aug": 2.9, "Sep": 2.8, "Oct": 1.9, "Nov": 1.1, "Dec": 1.1},
    "OHY": {"Jan": 4.0, "Feb": 4.3, "Mar": 4.9, "Apr": 3.5, "May": 3.3, "Jun": 3.1, "Jul": 3.2, "Aug": 3.4, "Sep": 3.9, "Oct": 4.2, "Nov": 3.8, "Dec": 4.0},
    "PAL": {"Jan": 3.9, "Feb": 4.7, "Mar": 4.2, "Apr": 3.9, "May": 3.9, "Jun": 3.9, "Jul": 3.7, "Aug": 4.1, "Sep": 3.4, "Oct": 3.7, "Nov": 3.8, "Dec": 4.5},
    "PAR": {"Jan": 4.9, "Feb": 5.8, "Mar": 5.8, "Apr": 6.2, "May": 5.7, "Jun": 3.4, "Jul": 3.6, "Aug": 3.4, "Sep": 3.7, "Oct": 4.0, "Nov": 4.1, "Dec": 4.7},
    "PAY": {"Jan": 2.0, "Feb": 3.6, "Mar": 3.7, "Apr": 3.4, "May": 3.2, "Jun": 3.3, "Jul": 3.2, "Aug": 3.2, "Sep": 3.0, "Oct": 3.0, "Nov": 2.8, "Dec": 2.7},
    "PSU": {"Jan": 2.6, "Feb": 2.7, "Mar": 2.7, "Apr": 2.9, "May": 3.0, "Jun": 3.7, "Jul": 3.8, "Aug": 3.9, "Sep": 2.9, "Oct": 2.7, "Nov": 2.5, "Dec": 2.5},
    "PTR": {"Jan": 3.2, "Feb": 3.1, "Mar": 3.1, "Apr": 3.0, "May": 2.9, "Jun": 2.8, "Jul": 2.8, "Aug": 2.9, "Sep": 3.0, "Oct": 3.2, "Nov": 3.3, "Dec": 3.5},
    "QIQ": {"Jan": 3.8, "Feb": 4.3, "Mar": 3.3, "Apr": 3.6, "May": 4.1, "Jun": 3.9, "Jul": 3.9, "Aug": 3.3, "Sep": 3.4, "Oct": 3.2, "Nov": 2.9, "Dec": 3.3},
    "REG": {"Jan": 2.9, "Feb": 4.7, "Mar": 3.1, "Apr": 3.3, "May": 3.4, "Jun": 3.0, "Jul": 2.9, "Aug": 3.2, "Sep": 3.0, "Oct": 2.6, "Nov": 2.5, "Dec": 2.6},
    "RLM": {"Jan": 4.5, "Feb": 2.6, "Mar": 4.0, "Apr": 3.5, "May": 3.2, "Jun": 3.3, "Jul": 3.5, "Aug": 5.1, "Sep": 7.1, "Oct": 6.5, "Nov": 5.1, "Dec": 4.0},
    "RUN": {"Jan": 3.9, "Feb": 4.0, "Mar": 3.7, "Apr": 3.6, "May": 3.7, "Jun": 3.6, "Jul": 3.6, "Aug": 3.6, "Sep": 3.7, "Oct": 3.9, "Nov": 3.9, "Dec": 3.9},
    "SAP": {"Jan": 3.2, "Feb": 3.6, "Mar": 4.3, "Apr": 5.3, "May": 5.1, "Jun": 4.0, "Jul": 3.9, "Aug": 3.2, "Sep": 2.9, "Oct": 3.3, "Nov": 3.5, "Dec": 3.0},
    "SBO": {"Jan": 3.8, "Feb": 3.9, "Mar": 4.5, "Apr": 5.4, "May": 5.2, "Jun": 3.8, "Jul": 4.1, "Aug": 4.1, "Sep": 4.1, "Oct": 4.5, "Nov": 3.8, "Dec": 4.1},
    "SEL": {"Jan": 3.6, "Feb": 3.7, "Mar": 4.1, "Apr": 5.2, "May": 6.2, "Jun": 4.6, "Jul": 4.6, "Aug": 4.6, "Sep": 4.9, "Oct": 4.5, "Nov": 4.0, "Dec": 3.7},
    "SMS": {"Jan": 2.9, "Feb": 2.9, "Mar": 2.9, "Apr": 2.7, "May": 2.7, "Jun": 2.8, "Jul": 2.7, "Aug": 3.0, "Sep": 3.7, "Oct": 3.1, "Nov": 3.1, "Dec": 2.9},
    "SON": {"Jan": 2.3, "Feb": 2.9, "Mar": 2.8, "Apr": 2.7, "May": 2.8, "Jun": 2.7, "Jul": 2.5, "Aug": 2.6, "Sep": 2.2, "Oct": 2.2, "Nov": 2.2, "Dec": 2.1},
    "SOV": {"Jan": 3.8, "Feb": 5.0, "Mar": 5.1, "Apr": 6.5, "May": 7.7, "Jun": 6.1, "Jul": 6.2, "Aug": 6.0, "Sep": 5.2, "Oct": 4.4, "Nov": 4.4, "Dec": 4.2},
    "SPO": {"Jan": 1.0, "Feb": 1.0, "Mar": 1.0, "Apr": 1.0, "May": 1.0, "Jun": 1.0, "Jul": 1.0, "Aug": 1.0, "Sep": 1.0, "Oct": 1.0, "Nov": 1.0, "Dec": 1.0},
    "SXF": {"Jan": 2.9, "Feb": 3.1, "Mar": 3.0, "Apr": 3.3, "May": 3.0, "Jun": 3.2, "Jul": 3.4, "Aug": 3.3, "Sep": 3.1, "Oct": 2.7, "Nov": 2.8, "Dec": 2.7},
    "SYO": {"Jan": 2.5, "Feb": 2.7, "Mar": 2.6, "Apr": 3.0, "May": 1.1, "Jun": 1.1, "Jul": 1.1, "Aug": 1.1, "Sep": 3.9, "Oct": 3.7, "Nov": 2.6, "Dec": 2.5},
    "TAM": {"Jan": 2.9, "Feb": 3.0, "Mar": 3.8, "Apr": 4.8, "May": 5.0, "Jun": 5.5, "Jul": 4.7, "Aug": 4.5, "Sep": 4.8, "Oct": 4.0, "Nov": 3.3, "Dec": 3.1},
    "TAT": {"Jan": 3.3, "Feb": 3.6, "Mar": 4.6, "Apr": 5.5, "May": 5.7, "Jun": 5.2, "Jul": 5.5, "Aug": 5.2, "Sep": 4.1, "Oct": 3.9, "Nov": 3.4, "Dec": 3.1},
    "TIK": {"Jan": 1.1, "Feb": 3.3, "Mar": 3.5, "Apr": 3.2, "May": 3.7, "Jun": 3.2, "Jul": 3.4, "Aug": 3.2, "Sep": 2.8, "Oct": 3.4, "Nov": 1.1, "Dec": 1.1},
    "TIR": {"Jan": 4.4, "Feb": 4.2, "Mar": 5.1, "Apr": 5.3, "May": 6.2, "Jun": 6.5, "Jul": 6.5, "Aug": 5.8, "Sep": 5.3, "Oct": 5.3, "Nov": 4.8, "Dec": 4.5},
    "TNB": {"Jan": 2.5, "Feb": 2.9, "Mar": 3.1, "Apr": 1.0, "May": 1.0, "Jun": 1.0, "Jul": 1.0, "Aug": 1.0, "Sep": 3.4, "Oct": 3.6, "Nov": 3.0, "Dec": 2.5},
    "TOR": {"Jan": 3.6, "Feb": 3.9, "Mar": 3.3, "Apr": 3.4, "May": 3.1, "Jun": 2.9, "Jul": 3.2, "Aug": 3.3, "Sep": 3.3, "Oct": 2.6, "Nov": 2.4, "Dec": 3.4},
    "XIA": {"Jan": 4.8, "Feb": 4.8, "Mar": 5.1, "Apr": 7.2, "May": 7.9, "Jun": 8.5, "Jul": 7.8, "Aug": 6.2, "Sep": 6.5, "Oct": 6.3, "Nov": 4.8, "Dec": 4.5},
    "YUS": {"Jan": 2.7, "Feb": 2.8, "Mar": 3.4, "Apr": 3.3, "May": 3.1, "Jun": 2.7, "Jul": 2.4, "Aug": 2.6, "Sep": 2.9, "Oct": 3.0, "Nov": 2.7, "Dec": 2.5},
}

"""
Solar constant (SC) [$E_{\text{sc}}$]. [W/m^2]
太阳常数 (SC) [$E_{\text{sc}}$]。[瓦/平方米]

References
----------
Gueymard, C. A. (2018). A reevaluation of the solar constant based on a 
42-year total solar irradiance time series and a reconciliation of 
spaceborne observations. Solar Energy, 168, 2-9.
"""
solar_constant = 1361.1 # temporary for testing


r"""
Stefan-Boltzmann constant [$\sigma_{\text{B}}$]. [W/m^2/K^4]
斯特芬-波兹曼常数 ($\sigma_{\text{B}}$)。[瓦/平方米/开尔文^4]
"""
sigmaB = 5.670374419e-8


"""
Engerer2 separation model parameters (C, b0, b1, b2, b3, b4, b5).
Engerer2 辐照分离模型参数 (C, b0, b1, b2, b3, b4, b5)。

Mapping by averaging period [minutes].
按平均时段 [分钟] 映射。

References
----------
Bright, J. M., & Engerer, N. A. (2019). Engerer2: Global 
re-parameterisation, update, and validation of an irradiance separation 
model at different temporal resolutions. Journal of Renewable and 
Sustainable Energy, 11(3), 033701.
"""
ENGERER2_PARAMS = {
    1: (0.105620, -4.13320, 8.25780, 0.0100870, 0.000888010, -4.93020, 0.443780),
    5: (0.0939360, -4.57710, 8.46410, 0.0100120, 0.00397500, -4.39210, 0.393310),
    10: (0.0799650, -4.85390, 8.47640, 0.0188490, 0.00514970, -4.14570, 0.374660),
    15: (0.0659720, -4.72110, 8.32940, 0.00954440, 0.00534930, -4.16900, 0.395260),
    30: (0.0326750, -4.86810, 8.18670, 0.0158290, 0.00599220, -4.03040, 0.473710),
    60: (-0.00975390, -5.31690, 8.50840, 0.0132410, 0.00743560, -3.03290, 0.564030),
    1440: (0.327260, -9.43910, 17.1130, 0.137520, -0.0240990, 6.62570, 0.314190),
}


"""
Yang4 separation model parameters.
Yang4 辐照分离模型参数。

Predictors: kt, AST, zenith, dktc, k_de, k_60min.
预测变量：kt, AST, zenith, dktc, k_de, k_60min。

References
----------
Yang, D., & Boland, J. (2019). Satellite-augmented diffuse solar 
radiation separation models. Journal of Renewable and Sustainable Energy, 
11(2), 023704.
"""
YANG4_PARAMS = (
    0.0361,   # C (lower bound) / C (下限)
    -0.5744,  # b0 (intercept) / b0 (截距)
    4.3184,   # b1 (kt) / b1 (晴朗指数)
    -0.0011,  # b2 (AST) / b2 (地面太阳时)
    0.0004,   # b3 (zenith) / b3 (太阳天顶角)
    -4.7952,  # b4 (dktc) / b4 (晴空偏离度)
    1.4414,   # b5 (k_de) / b5 (云增强分数)
    -2.8396,  # b6 (k_60min) / b6 (60 分钟平均散射分数)
)


"""
Hugging Face dataset repositories.
Hugging Face 数据集仓库。

Each satellite-derived or reanalysis product is stored as monthly parquet files
on Hugging Face. The repository IDs and maintainer emails are listed below.
每种卫星反演或再分析产品以月度 parquet 文件形式存储在 Hugging Face 上。
仓库 ID 与维护者邮箱如下。
"""
# Mainteiner email for HF updates / HF 更新维护者邮箱
HF_MAINTAINER_EMAIL = "yangdazhi.nus@gmail.com"

# MERRA-2 (REST2 clear-sky inputs) / MERRA-2（REST2 晴空输入）
MERRA2_HF_REPO_ID = "dazhiyang/bsrn-merra2"

# CAMS Radiation Service (CRS) / CAMS 辐射服务 (CRS)
CRS_HF_REPO_ID = "dazhiyang/bsrn-crs"

# NSRDB PSM v4 (one repo per variant) / NSRDB PSM v4（每变体一个仓库）
NSRDB_CONUS_HF_REPO_ID = "dazhiyang/bsrn-nsrdb-conus"
NSRDB_FULL_DISC_HF_REPO_ID = "dazhiyang/bsrn-nsrdb-full-disc"
NSRDB_AGGREGATED_HF_REPO_ID = "dazhiyang/bsrn-nsrdb-aggregated"


"""
Geostationary satellite subsatellite longitudes [degrees].
静止卫星星下点经度 [度]。

All geostationary satellites orbit at 0° latitude; the Earth-disk
reliability radius is approximately 60° for all entries.
所有静止卫星轨道纬度为 0°；地球圆盘可靠性半径约为 60°。

References
----------
.. [1] GOES-R Series. https://www.goes-r.gov
.. [2] JMA Himawari-8/9. https://www.data.jma.go.jp/mscweb/en/himawari89/
.. [3] EUMETSAT Meteosat. https://www.eumetsat.int
"""
GEO_SATELLITE_LON_DEG = {
    "MSG": 0.0,
    "Himawari": 140.7,
    "GOES-East": -75.2,
    "GOES-West": -137.0
}
GEO_SATELLITE_LAT_DEG = 0.0
GEO_SATELLITE_DISK_RADIUS_DEG = 60.0


"""
CAMS/McClear parsing constants.
CAMS/McClear 解析常量。

Base host for CAMS/McClear API (SoDa).
CAMS/McClear API（SoDa）的基础主机名。

Map CAMS/McClear variable names to project variable names.
将 CAMS/McClear 变量名映射到项目变量名。

References
----------
.. [1] Lefèvre, M., Oumbe, A., Blanc, P., Espinar, B., Gschwind, B., Qu, Z.,
       et al. (2013). McClear: A new model estimating downwelling solar
       radiation at ground level in clear-sky conditions. Atmospheric
       Measurement Techniques, 6(9), 2403–2418.
.. [2] Gschwind, B., Wald, L., Blanc, P., Lefèvre, M., Schroedter-Homscheidt, M.,
       & Arola, A. (2019). Improving the McClear model estimating the downwelling
       solar radiation at ground level in cloud-free conditions – McClear-v3.
       Meteorologische Zeitschrift, 28(2).
"""
MCCLEAR_API_HOST = "api.soda-solardata.com"
MCCLEAR_INTEGRATED_COLUMNS = [
    "TOA", "Clear sky GHI", "Clear sky BHI", "Clear sky DHI", "Clear sky BNI",
    "GHI", "BHI", "DHI", "BNI", "GHI no corr", "BHI no corr", "DHI no corr", "BNI no corr",
]
MCCLEAR_VARIABLE_MAP = {
    "TOA": "ghi_extra",
    "Clear sky GHI": "ghi_clear",
    "Clear sky BHI": "bhi_clear",
    "Clear sky DHI": "dhi_clear",
    "Clear sky BNI": "bni_clear",
    "GHI": "ghi",
    "BHI": "bhi",
    "DHI": "dhi",
    "BNI": "bni",
    "sza": "zenith",
}

"""
CAMS Radiation Service (CRS) on SoDa.
SoDa 上的 CAMS 辐射服务（CRS）。

Same HTTPS host and CSV layout as McClear; WPS ``Identifier`` is ``get_cams_radiation``.
与 McClear 使用相同 HTTPS 主机与 CSV 布局；WPS ``Identifier`` 为 ``get_cams_radiation``。

SoDa CRS CSV also includes clear-sky columns, ``TOA``, ``BHI``, ``Reliability``, etc.
:func:`bsrn.io.crs.download_crs` keeps only all-sky ``GHI``, ``BNI``, ``DHI`` (Wh→W) as
``ghi_crs``, ``bni_crs``, ``dhi_crs`` plus the UTC time index.
SoDa CRS CSV 另含晴空分量、``TOA``、``BHI``、``Reliability`` 等；:func:`bsrn.io.crs.download_crs`
仅保留全天空 ``GHI``、``BNI``、``DHI``（Wh→W 换算后列名为 ``ghi_crs`` 等）及 UTC 时间索引。

Himawari and MSG GEO metadata (subsatellite point, **60°** reliability radius, minimum
*start* dates) used by :func:`bsrn.io.crs.download_crs` via ``_check_crs_coverage``.
Himawari 与 MSG 的 GEO 元数据（星下点、60° 可靠性半径、最早起始日），由 ``_check_crs_coverage`` 使用。

All-sky columns are named ``ghi_crs``, ``bni_crs``, ``dhi_crs`` to leave room for other
gridded products (e.g. ``ghi_nsrdb``) when merging workflows.
全天空列使用 ``*_crs`` 后缀，便于与后续其他格点产品（如 ``ghi_nsrdb``）并列区分。

References
----------
.. [1] Schroedter-Homscheidt, M., et al. (2016). User’s Guide to the CAMS Radiation Service.
       European Commission.
.. [2] CAMS radiation service — SoDa. 
       https://www.soda-pro.com/web-services/radiation/cams-radiation-service
"""
CRS_MSG_MIN_START_UTC = "2004-01-01"
CRS_HIMAWARI_MIN_START_UTC = "2016-01-01"

CRS_API_HOST = MCCLEAR_API_HOST
CRS_INTEGRATED_COLUMNS = ["GHI", "BNI", "DHI"]
CRS_VARIABLE_MAP = {
    "GHI": "ghi_crs",
    "BNI": "bni_crs",
    "DHI": "dhi_crs",
}
CRS_OUTPUT_COLUMNS = ["ghi_crs", "bni_crs", "dhi_crs"]


"""
NSRDB (National Solar Radiation Database) on NLR.
NLR 上的 NSRDB (国家太阳辐射数据库)。

Supports three variants:
1. conus (v4.0.0): 5-min, 2018–2024, CONUS (GOES-East + GOES-West overlap).
2. full-disc (v4.0.0): 10-min, 2018–2024, full GOES Earth disk.
3. aggregated (v4.0.0): 30-min, 1998–2024, aggregated GOES archive.

References
----------
.. [1] Sengupta, M., Xie, Y., Lopez, A., Habte, A., Maclaurin, G., & Shelby, J. 
       (2018). The national solar radiation data base (NSRDB). Renewable 
       and sustainable energy reviews, 89, 51-60.
.. [2] NSRDB Physical Solar Model (PSM) v4.0.0.
       https://nsrdb.nrel.gov/data-viewer
"""
NSRDB_API_BASE_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/"

NSRDB_VARIANTS = {
    "conus": {
        "endpoint": "nsrdb-GOES-conus-v4-0-0-download.csv",
        "interval": 5,
        "years": (2018, 2024),
        "satellites": ["GOES-East", "GOES-West"],
        "bbox": {"lat": (20, 50), "lon": (-130, -66)},
        "folder": "nsrdb-conus",
        "hf_repo_id": NSRDB_CONUS_HF_REPO_ID,
    },
    "full-disc": {
        "endpoint": "nsrdb-GOES-full-disc-v4-0-0-download.csv",
        "interval": 10,
        "years": (2018, 2024),
        "satellites": ["GOES-East", "GOES-West"],
        "folder": "nsrdb-full-disc",
        "hf_repo_id": NSRDB_FULL_DISC_HF_REPO_ID,
    },
    "aggregated": {
        "endpoint": "nsrdb-GOES-aggregated-v4-0-0-download.csv",
        "interval": 30,
        "years": (1998, 2024),
        "satellites": ["GOES-East", "GOES-West"],
        "folder": "nsrdb-aggregated",
        "hf_repo_id": NSRDB_AGGREGATED_HF_REPO_ID,
    },
}

NSRDB_VARIABLE_MAP = {
    "GHI": "ghi_nsrdb",
    "DNI": "bni_nsrdb",
    "DHI": "dhi_nsrdb",
}
NSRDB_OUTPUT_COLUMNS = ["ghi_nsrdb", "bni_nsrdb", "dhi_nsrdb"]

"""
Wong colorblind-friendly palette.
Wong 色盲友好配色方案。

References
----------
Wong, B. (2011). Points of view: Color blindness. Nature Methods, 
8(6), 441-441.
"""
WONG_PALETTE = [
    "#E69F00",  # Orange / 橙色
    "#56B4E9",  # Sky Blue / 天蓝色
    "#009E73",  # Bluish Green / 青绿色
    "#CC79A7",  # Reddish Purple / 浅紫红色
    "#D55E00",  # Vermillion / 朱红色
    "#F0E442",  # Yellow / 黄色
    "#0072B2"   # Blue / 蓝色
]
