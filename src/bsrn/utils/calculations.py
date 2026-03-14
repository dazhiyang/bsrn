"""
Supporting mathematical and radiometric calculations.
辅助数学和辐射度计算。
"""

import numpy as np


def calc_kt(ghi, ghi_extra):
    """
    Calculates clearness index ($k_t$).
    计算晴朗指数 ($k_t$)。

    Parameters
    ----------
    ghi : numeric or Series
        Measured global horizontal irradiance ($G_h$). [W/m^2]
        测量的水平总辐照度 ($G_h$)。[瓦/平方米]
    ghi_extra : numeric or Series
        Extraterrestrial horizontal irradiance ($E_0$). [W/m^2]
        地外水平辐照度 ($E_0$)。[瓦/平方米]

    Returns
    -------
    kt : numeric or Series
        Clearness index ($k_t = G_h / E_0$). [unitless]
        晴朗指数 ($k_t = G_h / E_0$)。[无单位]
    """
    return ghi / np.maximum(ghi_extra, 0.01)


def calc_kb(bni, bni_extra):
    """
    Calculates beam transmittance ($k_b$).
    计算直射透射率 ($k_b$)。

    Parameters
    ----------
    bni : numeric or Series
        Measured beam normal irradiance ($B_n$). [W/m^2]
        测量的法向直接辐照度 ($B_n$)。[瓦/平方米]
    bni_extra : numeric or Series
        Extraterrestrial beam normal irradiance ($E_{0n}$). [W/m^2]
        地外法向辐照度 ($E_{0n}$)。[瓦/平方米]

    Returns
    -------
    kb : numeric or Series
        Beam transmittance ($k_b = B_n / E_{0n}$). [unitless]
        直射透射率 ($k_b = B_n / E_{0n}$)。[无单位]
    """
    return bni / np.maximum(bni_extra, 0.01)


def calc_kd(dhi, ghi_extra):
    """
    Calculates diffuse transmittance ($k_d$).
    计算散射透射率 ($k_d$)。

    Parameters
    ----------
    dhi : numeric or Series
        Measured diffuse horizontal irradiance ($D_h$). [W/m^2]
        测量的水平散射辐照度 ($D_h$)。[瓦/平方米]
    ghi_extra : numeric or Series
        Extraterrestrial horizontal irradiance ($E_0$). [W/m^2]
        地外水平辐照度 ($E_0$)。[瓦/平方米]

    Returns
    -------
    kd : numeric or Series
        Diffuse transmittance ($k_d = D_h / E_0$). [unitless]
        散射透射率 ($k_d = D_h / E_0$)。[无单位]
    """
    return dhi / np.maximum(ghi_extra, 0.01)


def calc_k(dhi, ghi):
    """
    Calculates diffuse fraction ($k$).
    计算散射分数 ($k$)。

    Parameters
    ----------
    dhi : numeric or Series
        Measured diffuse horizontal irradiance ($D_h$). [W/m^2]
        测量的水平散射辐照度 ($D_h$)。[瓦/平方米]
    ghi : numeric or Series
        Measured global horizontal irradiance ($G_h$). [W/m^2]
        测量的水平总辐照度 ($G_h$)。[瓦/平方米]

    Returns
    -------
    k : numeric or Series
        Diffuse fraction ($k = D_h / G_h$). [unitless]
        散射分数 ($k = D_h / G_h$)。[无单位]
    """
    return dhi / np.maximum(ghi, 0.01)
