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
        Measured global horizontal irradiance ($G_h$).
        测量的水平总辐照度 ($G_h$)。
    ghi_extra : numeric or Series
        Extraterrestrial horizontal irradiance ($E_0$).
        地外水平辐照度 ($E_0$)。

    Returns
    -------
    kt : numeric or Series
        Clearness index ($k_t = G_h / E_0$).
        晴朗指数 ($k_t = G_h / E_0$)。
    """
    return ghi / np.maximum(ghi_extra, 0.01)


def calc_kb(bni, bni_extra):
    """
    Calculates beam transmittance ($k_b$).
    计算直射透射率 ($k_b$)。

    Parameters
    ----------
    bni : numeric or Series
        Measured beam normal irradiance ($B_n$).
        测量的法向直接辐照度 ($B_n$)。
    bni_extra : numeric or Series
        Extraterrestrial beam normal irradiance ($E_{0n}$).
        地外法向辐照度 ($E_{0n}$)。

    Returns
    -------
    kb : numeric or Series
        Beam transmittance ($k_b = B_n / E_{0n}$).
        直射透射率 ($k_b = B_n / E_{0n}$)。
    """
    return bni / np.maximum(bni_extra, 0.01)


def calc_kd(dhi, ghi_extra):
    """
    Calculates diffuse transmittance ($k_d$).
    计算散射透射率 ($k_d$)。

    Parameters
    ----------
    dhi : numeric or Series
        Measured diffuse horizontal irradiance ($D_h$).
        测量的水平散射辐照度 ($D_h$)。
    ghi_extra : numeric or Series
        Extraterrestrial horizontal irradiance ($E_0$).
        地外水平辐照度 ($E_0$)。

    Returns
    -------
    kd : numeric or Series
        Diffuse transmittance ($k_d = D_h / E_0$).
        散射透射率 ($k_d = D_h / E_0$)。
    """
    return dhi / np.maximum(ghi_extra, 0.01)


def calc_k(dhi, ghi):
    """
    Calculates diffuse fraction ($k$).
    计算散射分数 ($k$)。

    Parameters
    ----------
    dhi : numeric or Series
        Measured diffuse horizontal irradiance ($D_h$).
        测量的水平散射辐照度 ($D_h$)。
    ghi : numeric or Series
        Measured global horizontal irradiance ($G_h$).
        测量的水平总辐照度 ($G_h$)。

    Returns
    -------
    k : numeric or Series
        Diffuse fraction ($k = D_h / G_h$).
        散射分数 ($k = D_h / G_h$)。
    """
    return dhi / np.maximum(ghi, 0.01)
