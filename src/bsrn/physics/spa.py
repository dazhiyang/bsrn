"""
Solar Position Algorithm (SPA) for solar radiation applications.
Ported from pvlib-python (Holmgren et al. 2018).
太阳位置算法 (SPA)。
从 pvlib-python 移植 (Holmgren et al. 2018)。

References:
[1] Reda, I.; Andreas, A. (2004). Solar Position Algorithm for Solar Radiation 
Applications. Solar Energy. 76 (5): 577-589.
[2] Holmgren, William F., Clifford W. Hansen, and Mark A. Mikofski. "pvlib python: 
A python package for modeling solar energy systems." Journal of Open Source Software 
3.29 (2018): 884.
[3] Anderson, Kevin S., et al. "pvlib python: 2023 project update." Journal of Open 
Source Software 8.92 (2023): 5994.
"""

import numpy as np
import pandas as pd

# heliocentric longitude coefficients / 日心经度系数
L0 = np.array([
    [175347046.0, 0.0, 0.0], [3341656.0, 4.6692568, 6283.07585],
    [34894.0, 4.6261, 12566.1517], [3497.0, 2.7441, 5753.3849],
    [3418.0, 2.8289, 3.5231], [3136.0, 3.6277, 77713.7715],
    [2676.0, 4.4181, 7860.4194], [2343.0, 6.1352, 3930.2097],
    [1324.0, 0.7425, 11506.7698], [1273.0, 2.0371, 529.691],
    [1199.0, 1.1096, 1577.3435], [990.0, 5.233, 5884.927],
    [902.0, 2.045, 26.298], [857.0, 3.508, 398.149],
    [780.0, 1.179, 5223.694], [753.0, 2.533, 5507.553],
    [505.0, 4.583, 18849.228], [492.0, 4.205, 775.523],
    [357.0, 2.92, 0.067], [317.0, 5.849, 11790.629],
    [284.0, 1.899, 796.298], [271.0, 0.315, 10977.079],
    [243.0, 0.345, 5486.778], [206.0, 4.806, 2544.314],
    [205.0, 1.869, 5573.143], [202.0, 2.458, 6069.777],
    [156.0, 0.833, 213.299], [132.0, 3.411, 2942.463],
    [126.0, 1.083, 20.775], [115.0, 0.645, 0.98],
    [103.0, 0.636, 4694.003], [102.0, 0.976, 15720.839],
    [102.0, 4.267, 7.114], [99.0, 6.21, 2146.17],
    [98.0, 0.68, 155.42], [86.0, 5.98, 161000.69],
    [85.0, 1.3, 6275.96], [85.0, 3.67, 71430.7],
    [80.0, 1.81, 17260.15], [79.0, 3.04, 12036.46],
    [75.0, 1.76, 5088.63], [74.0, 3.5, 3154.69],
    [74.0, 4.68, 801.82], [70.0, 0.83, 9437.76],
    [62.0, 3.98, 8827.39], [61.0, 1.82, 7084.9],
    [57.0, 2.78, 6286.6], [56.0, 4.39, 14143.5],
    [56.0, 3.47, 6279.55], [52.0, 0.19, 12139.55],
    [52.0, 1.33, 1748.02], [51.0, 0.28, 5856.48],
    [49.0, 0.49, 1194.45], [41.0, 5.37, 8429.24],
    [41.0, 2.4, 19651.05], [39.0, 6.17, 10447.39],
    [37.0, 6.04, 10213.29], [37.0, 2.57, 1059.38],
    [36.0, 1.71, 2352.87], [36.0, 1.78, 6812.77],
    [33.0, 0.59, 17789.85], [30.0, 0.44, 83996.85],
    [30.0, 2.74, 1349.87], [25.0, 3.16, 4690.48]
])
L1 = np.array([
    [628331966747.0, 0.0, 0.0], [206059.0, 2.678235, 6283.07585],
    [4303.0, 2.6351, 12566.1517], [425.0, 1.59, 3.523],
    [119.0, 5.796, 26.298], [109.0, 2.966, 1577.344],
    [93.0, 2.59, 18849.23], [72.0, 1.14, 529.69],
    [68.0, 1.87, 398.15], [67.0, 4.41, 5507.55],
    [59.0, 2.89, 5223.69], [56.0, 2.17, 155.42],
    [45.0, 0.4, 796.3], [36.0, 0.47, 775.52],
    [29.0, 2.65, 7.11], [21.0, 5.34, 0.98],
    [19.0, 1.85, 5486.78], [19.0, 4.97, 213.3],
    [17.0, 2.99, 6275.96], [16.0, 0.03, 2544.31],
    [16.0, 1.43, 2146.17], [15.0, 1.21, 10977.08],
    [12.0, 2.83, 1748.02], [12.0, 3.26, 5088.63],
    [12.0, 5.27, 1194.45], [12.0, 2.08, 4694.0],
    [11.0, 0.77, 553.57], [10.0, 1.3, 6286.6],
    [10.0, 4.24, 1349.87], [9.0, 2.7, 242.73],
    [9.0, 5.64, 951.72], [8.0, 5.3, 2352.87],
    [6.0, 2.65, 9437.76], [6.0, 4.67, 4690.48]
])
L2 = np.array([
    [52919.0, 0.0, 0.0], [8720.0, 1.0721, 6283.0758],
    [309.0, 0.867, 12566.152], [27.0, 0.05, 3.52],
    [16.0, 5.19, 26.3], [16.0, 3.68, 155.42],
    [10.0, 0.76, 18849.23], [9.0, 2.06, 77713.77],
    [7.0, 0.83, 775.52], [5.0, 4.66, 1577.34],
    [4.0, 1.03, 7.11], [4.0, 3.44, 5573.14],
    [3.0, 5.14, 796.3], [3.0, 6.05, 5507.55],
    [3.0, 1.19, 242.73], [3.0, 6.12, 529.69],
    [3.0, 0.31, 398.15], [3.0, 2.28, 553.57],
    [2.0, 4.38, 5223.69], [2.0, 3.75, 0.98]
])
L3 = np.array([
    [289.0, 5.844, 6283.076], [35.0, 0.0, 0.0],
    [17.0, 5.49, 12566.15], [3.0, 5.2, 155.42],
    [1.0, 4.72, 3.52], [1.0, 5.3, 18849.23],
    [1.0, 5.97, 242.73]
])
L4 = np.array([[114.0, 3.142, 0.0], [8.0, 4.13, 6283.08], [1.0, 3.84, 12566.15]])
L5 = np.array([[1.0, 3.14, 0.0]])

# heliocentric latitude coefficients / 日心纬度系数
B0 = np.array([
    [280.0, 3.199, 84334.662], [102.0, 5.422, 5507.553],
    [80.0, 3.88, 5223.69], [44.0, 3.7, 2352.87], [32.0, 4.0, 1577.34]
])
B1 = np.array([[9.0, 3.9, 5507.55], [6.0, 1.73, 5223.69]])

# heliocentric radius coefficients / 日心半径系数
R0 = np.array([
    [100013989.0, 0.0, 0.0], [1670700.0, 3.0984635, 6283.07585],
    [13956.0, 3.05525, 12566.1517], [3084.0, 5.1985, 77713.7715],
    [1628.0, 1.1739, 5753.3849], [1576.0, 2.8469, 7860.4194],
    [925.0, 5.453, 11506.77], [542.0, 4.564, 3930.21],
    [472.0, 3.661, 5884.927], [346.0, 0.964, 5507.553],
    [329.0, 5.9, 5223.694], [307.0, 0.299, 5573.143],
    [243.0, 4.273, 11790.629], [212.0, 5.847, 1577.344],
    [186.0, 5.022, 10977.079], [175.0, 3.012, 18849.228],
    [110.0, 5.055, 5486.778], [98.0, 0.89, 6069.78],
    [86.0, 5.69, 15720.84], [86.0, 1.27, 161000.69],
    [65.0, 0.27, 17260.15], [63.0, 0.92, 529.69],
    [57.0, 2.01, 83996.85], [56.0, 5.24, 71430.7],
    [49.0, 3.25, 2544.31], [47.0, 2.58, 775.52],
    [45.0, 5.54, 9437.76], [43.0, 6.01, 6275.96],
    [39.0, 5.36, 4694.0], [38.0, 2.39, 8827.39],
    [37.0, 0.83, 19651.05], [37.0, 4.9, 12139.55],
    [36.0, 1.67, 12036.46], [35.0, 1.84, 2942.46],
    [33.0, 0.24, 7084.9], [32.0, 0.18, 5088.63],
    [32.0, 1.78, 398.15], [28.0, 1.21, 6286.6],
    [28.0, 1.9, 6279.55], [26.0, 4.59, 10447.39]
])
R1 = np.array([
    [103019.0, 1.10749, 6283.07585], [1721.0, 1.0644, 12566.1517],
    [702.0, 3.142, 0.0], [32.0, 1.02, 18849.23],
    [31.0, 2.84, 5507.55], [25.0, 1.32, 5223.69],
    [18.0, 1.42, 1577.34], [10.0, 5.91, 10977.08],
    [9.0, 1.42, 6275.96], [9.0, 0.27, 5486.78]
])
R2 = np.array([
    [4359.0, 5.7846, 6283.0758], [124.0, 5.579, 12566.152],
    [12.0, 3.14, 0.0], [9.0, 3.63, 77713.77],
    [6.0, 1.87, 5573.14], [3.0, 5.47, 18849.23]
])
R3 = np.array([[145.0, 4.273, 6283.076], [7.0, 3.92, 12566.15]])
R4 = np.array([[4.0, 2.56, 6283.08]])

NUTATION_ABCD_ARRAY = np.array([
    [-171996, -174.2, 92025, 8.9], [-13187, -1.6, 5736, -3.1],
    [-2274, -0.2, 977, -0.5], [2062, 0.2, -895, 0.5],
    [1426, -3.4, 54, -0.1], [712, 0.1, -7, 0],
    [-517, 1.2, 224, -0.6], [-386, -0.4, 200, 0],
    [-301, 0, 129, -0.1], [217, -0.5, -95, 0.3],
    [-158, 0, 0, 0], [129, 0.1, -70, 0],
    [123, 0, -53, 0], [63, 0, 0, 0],
    [63, 0.1, -33, 0], [-59, 0, 26, 0],
    [-58, -0.1, 32, 0], [-51, 0, 27, 0],
    [48, 0, 0, 0], [46, 0, -24, 0],
    [-38, 0, 16, 0], [-31, 0, 13, 0],
    [29, 0, 0, 0], [29, 0, -12, 0],
    [26, 0, 0, 0], [-22, 0, 0, 0],
    [21, 0, -10, 0], [17, -0.1, 0, 0],
    [16, 0, -8, 0], [-16, 0.1, 7, 0],
    [-15, 0, 9, 0], [-13, 0, 7, 0],
    [-12, 0, 6, 0], [11, 0, 0, 0],
    [-10, 0, 5, 0], [-8, 0, 3, 0],
    [7, 0, -3, 0], [-7, 0, 0, 0],
    [-7, 0, 3, 0], [-7, 0, 3, 0],
    [6, 0, 0, 0], [6, 0, -3, 0],
    [6, 0, -3, 0], [-6, 0, 3, 0],
    [-6, 0, 3, 0], [5, 0, 0, 0],
    [-5, 0, 3, 0], [-5, 0, 3, 0],
    [-5, 0, 3, 0], [4, 0, 0, 0],
    [4, 0, 0, 0], [4, 0, 0, 0],
    [-4, 0, 0, 0], [-4, 0, 0, 0],
    [-4, 0, 0, 0], [3, 0, 0, 0],
    [-3, 0, 0, 0], [-3, 0, 0, 0],
    [-3, 0, 0, 0], [-3, 0, 0, 0],
    [-3, 0, 0, 0], [-3, 0, 0, 0],
    [-3, 0, 0, 0]
])

NUTATION_YTERM_ARRAY = np.array([
    [0, 0, 0, 0, 1], [-2, 0, 0, 2, 2], [0, 0, 0, 2, 2], [0, 0, 0, 0, 2],
    [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [-2, 1, 0, 2, 2], [0, 0, 0, 2, 1],
    [0, 0, 1, 2, 2], [-2, -1, 0, 2, 2], [-2, 0, 1, 0, 0], [-2, 0, 0, 2, 1],
    [0, 0, -1, 2, 2], [2, 0, 0, 0, 0], [0, 0, 1, 0, 1], [2, 0, -1, 2, 2],
    [0, 0, -1, 0, 1], [0, 0, 1, 2, 1], [-2, 0, 2, 0, 0], [0, 0, -2, 2, 1],
    [2, 0, 0, 2, 2], [0, 0, 2, 2, 2], [0, 0, 2, 0, 0], [-2, 0, 1, 2, 2],
    [0, 0, 0, 2, 0], [-2, 0, 0, 2, 0], [0, 0, -1, 2, 1], [0, 2, 0, 0, 0],
    [2, 0, -1, 0, 1], [-2, 2, 0, 2, 2], [0, 1, 0, 0, 1], [-2, 0, 1, 0, 1],
    [0, -1, 0, 0, 1], [0, 0, 2, -2, 0], [2, 0, -1, 2, 1], [2, 0, 1, 2, 2],
    [0, 1, 0, 2, 2], [-2, 1, 1, 0, 0], [0, -1, 0, 2, 2], [2, 0, 0, 2, 1],
    [2, 0, 1, 0, 0], [-2, 0, 2, 2, 2], [-2, 0, 1, 2, 1], [2, 0, -2, 0, 1],
    [2, 0, 0, 0, 1], [0, -1, 1, 0, 0], [-2, -1, 0, 2, 1], [-2, 0, 0, 0, 1],
    [0, 0, 2, 2, 1], [-2, 0, 2, 0, 1], [-2, 1, 0, 2, 1], [0, 0, 1, -2, 0],
    [-1, 0, 1, 0, 0], [-2, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 2, 0],
    [0, 0, -2, 2, 2], [-1, -1, 1, 0, 0], [0, 1, 1, 0, 0], [0, -1, 1, 2, 2],
    [2, -1, -1, 2, 2], [0, 0, 3, 2, 2], [2, -1, 0, 2, 2]
])

def _julian_day(unixtime):
    return unixtime / 86400.0 + 2440587.5

def _sum_mult_cos_add_mult(arr, x):
    angles = arr[:, 1:2] + arr[:, 2:3] * x.reshape(1, -1)
    return np.sum(arr[:, 0:1] * np.cos(angles), axis=0)

def _mean_elongation(jce):
    return (297.85036 + 445267.111480 * jce - 0.0019142 * jce**2 + jce**3 / 189474)

def _mean_anomaly_sun(jce):
    return (357.52772 + 35999.050340 * jce - 0.0001603 * jce**2 - jce**3 / 300000)

def _mean_anomaly_moon(jce):
    return (134.96298 + 477198.867398 * jce + 0.0086972 * jce**2 + jce**3 / 56250)

def _moon_argument_latitude(jce):
    return (93.27191 + 483202.017538 * jce - 0.0036825 * jce**2 + jce**3 / 327270)

def _moon_ascending_longitude(jce):
    return (125.04452 - 1934.136261 * jce + 0.0020708 * jce**2 + jce**3 / 450000)

def _longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4):
    args = np.radians(
        NUTATION_YTERM_ARRAY[:, 0:1] * x0 + NUTATION_YTERM_ARRAY[:, 1:2] * x1 +
        NUTATION_YTERM_ARRAY[:, 2:3] * x2 + NUTATION_YTERM_ARRAY[:, 3:4] * x3 +
        NUTATION_YTERM_ARRAY[:, 4:5] * x4
    )
    delta_psi_sum = np.sum((NUTATION_ABCD_ARRAY[:, 0:1] + NUTATION_ABCD_ARRAY[:, 1:2] * jce) * np.sin(args), axis=0)
    delta_eps_sum = np.sum((NUTATION_ABCD_ARRAY[:, 2:3] + NUTATION_ABCD_ARRAY[:, 3:4] * jce) * np.cos(args), axis=0)
    return delta_psi_sum / 36000000.0, delta_eps_sum / 36000000.0

def _mean_ecliptic_obliquity(jme):
    U = jme / 10.0
    return (84381.448 - 4680.93 * U - 1.55 * U**2 + 1999.25 * U**3 - 51.38 * U**4 - 
            249.67 * U**5 - 39.05 * U**6 + 7.12 * U**7 + 27.87 * U**8 + 
            5.79 * U**9 + 2.45 * U**10)

def _true_ecliptic_obliquity(e0, deleps):
    return e0 / 3600.0 + deleps

def _aberration_correction(R):
    return -20.4898 / (3600.0 * R)

def _apparent_sun_longitude(theta, delta_psi, delta_tau):
    return theta + delta_psi + delta_tau

def _mean_sidereal_time(jd, jc):
    return (280.46061837 + 360.98564736629 * (jd - 2451545) + 
            0.000387933 * jc**2 - jc**3 / 38710000.0) % 360

def _apparent_sidereal_time(v0, delta_psi, epsilon):
    return v0 + delta_psi * np.cos(np.radians(epsilon))

def _geocentric_sun_right_ascension(lamd, epsilon, beta):
    eps_rad = np.radians(epsilon)
    lamd_rad = np.radians(lamd)
    beta_rad = np.radians(beta)
    num = np.sin(lamd_rad) * np.cos(eps_rad) - np.tan(beta_rad) * np.sin(eps_rad)
    alpha = np.degrees(np.arctan2(num, np.cos(lamd_rad)))
    return alpha % 360

def _geocentric_sun_declination(lamd, epsilon, beta):
    eps_rad = np.radians(epsilon)
    lamd_rad = np.radians(lamd)
    beta_rad = np.radians(beta)
    delta = np.degrees(np.arcsin(np.sin(beta_rad) * np.cos(eps_rad) + 
                                np.cos(beta_rad) * np.sin(eps_rad) * np.sin(lamd_rad)))
    return delta

def _local_hour_angle(v, lon, alpha):
    return (v + lon - alpha) % 360

def _equatorial_horizontal_parallax(r):
    return 8.794 / (3600.0 * r)

def _uterm(lat):
    return np.arctan(0.99664719 * np.tan(np.radians(lat)))

def _xterm(u, lat, elev):
    return np.cos(u) + (elev / 6378140.0) * np.cos(np.radians(lat))

def _yterm(u, lat, elev):
    return 0.99664719 * np.sin(u) + (elev / 6378140.0) * np.sin(np.radians(lat))

def _parallax_sun_right_ascension(x, eq_hor_par, h, delta):
    eq_hor_par_rad = np.radians(eq_hor_par)
    h_rad = np.radians(h)
    delta_rad = np.radians(delta)
    num = -x * np.sin(eq_hor_par_rad) * np.sin(h_rad)
    denom = np.cos(delta_rad) - x * np.sin(eq_hor_par_rad) * np.cos(h_rad)
    return np.degrees(np.arctan2(num, denom))

def _topocentric_sun_right_ascension(alpha, delta_alpha):
    return alpha + delta_alpha

def _topocentric_sun_declination(delta, x, y, eq_hor_par, delta_alpha, h):
    delta_rad = np.radians(delta)
    eq_hor_par_rad = np.radians(eq_hor_par)
    delta_alpha_rad = np.radians(delta_alpha)
    h_rad = np.radians(h)
    
    num = (np.sin(delta_rad) - y * np.sin(eq_hor_par_rad)) * np.cos(delta_alpha_rad)
    denom = np.cos(delta_rad) - x * np.sin(eq_hor_par_rad) * np.cos(h_rad)
    return np.degrees(np.arctan2(num, denom))

def _topocentric_local_hour_angle(h, delta_alpha):
    return h - delta_alpha

def _atmospheric_refraction_correction(pressure, temp, e0, atmos_refract=0.5667):
    """
    Atmospheric refraction correction per NREL SPA (Reda & Andreas, 2004).
    大气折射校正，基于 NREL SPA (Reda & Andreas, 2004)。
    """
    # switch: only apply when the sun is above the refraction-adjusted horizon
    switch = np.where(e0 >= -1.0 * (0.26667 + atmos_refract), 1.0, 0.0)
    delta_e = ((pressure / 1010.0) * (283.0 / (273.0 + temp))
               * 1.02 / (60.0 * np.tan(np.radians(
                   e0 + 10.3 / (e0 + 5.11))))) * switch
    return delta_e

def _solar_position(unixtime, lat, lon, elev, pressure=1013.25, temp=12, delta_t=69.0):
    jd = _julian_day(unixtime)
    jde = jd + delta_t / 86400.0
    jc = (jd - 2451545) / 36525.0
    jce = (jde - 2451545) / 36525.0
    jme = jce / 10.0
    
    R = (_sum_mult_cos_add_mult(R0, jme) + _sum_mult_cos_add_mult(R1, jme) * jme +
         _sum_mult_cos_add_mult(R2, jme) * jme**2 + _sum_mult_cos_add_mult(R3, jme) * jme**3 +
         _sum_mult_cos_add_mult(R4, jme) * jme**4) / 1e8
    
    L = (_sum_mult_cos_add_mult(L0, jme) + _sum_mult_cos_add_mult(L1, jme) * jme +
         _sum_mult_cos_add_mult(L2, jme) * jme**2 + _sum_mult_cos_add_mult(L3, jme) * jme**3 +
         _sum_mult_cos_add_mult(L4, jme) * jme**4 + _sum_mult_cos_add_mult(L5, jme) * jme**5) / 1e8
    L = np.rad2deg(L) % 360
    
    B = (_sum_mult_cos_add_mult(B0, jme) + _sum_mult_cos_add_mult(B1, jme) * jme) / 1e8
    B = np.rad2deg(B)
    
    theta = (L + 180.0) % 360
    beta = -B
    
    x0, x1, x2, x3, x4 = _mean_elongation(jce), _mean_anomaly_sun(jce), \
                         _mean_anomaly_moon(jce), _moon_argument_latitude(jce), \
                         _moon_ascending_longitude(jce)
    
    delta_psi, delta_epsilon = _longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4)
    epsilon0 = _mean_ecliptic_obliquity(jme)
    epsilon = _true_ecliptic_obliquity(epsilon0, delta_epsilon)
    delta_tau = _aberration_correction(R)
    lamd = _apparent_sun_longitude(theta, delta_psi, delta_tau)
    v0 = _mean_sidereal_time(jd, jc)
    v = _apparent_sidereal_time(v0, delta_psi, epsilon)
    alpha = _geocentric_sun_right_ascension(lamd, epsilon, beta)
    delta = _geocentric_sun_declination(lamd, epsilon, beta)
    H = _local_hour_angle(v, lon, alpha)
    # Topocentric conversions / 站心坐标转换 (Parallax correction)
    eq_hor_par = _equatorial_horizontal_parallax(R)
    u = _uterm(lat)
    x = _xterm(u, lat, elev)
    y = _yterm(u, lat, elev)
    
    delta_alpha = _parallax_sun_right_ascension(x, eq_hor_par, H, delta)
    
    alpha_prime = _topocentric_sun_right_ascension(alpha, delta_alpha)
    delta_prime = _topocentric_sun_declination(delta, x, y, eq_hor_par, delta_alpha, H)
    H_prime = _topocentric_local_hour_angle(H, delta_alpha)
    
    LatR = np.radians(lat)
    HR = np.radians(H_prime)
    DR = np.radians(delta_prime)
    
    e0 = np.degrees(np.arcsin(np.sin(LatR) * np.sin(DR) + np.cos(LatR) * np.cos(DR) * np.cos(HR)))
    delta_e = _atmospheric_refraction_correction(pressure, temp, e0)
    e = e0 + delta_e
    
    zenith_true = 90.0 - e0
    zenith_apparent = 90.0 - e
    
    # Azimuth / 方位角 (Measured from North)
    gamma = np.degrees(np.arctan2(np.sin(HR), np.cos(HR) * np.sin(LatR) - np.tan(DR) * np.cos(LatR)))
    azimuth = (gamma + 180.0) % 360
    
    return zenith_true, zenith_apparent, azimuth, R # zenith, apparent_zenith, azimuth, distance
