# src/biooptics/models/scattering.py

import numpy as np

def rayleigh_scattering_cross_section(wavelength_m: float, n: float) -> float:
    """
    计算 Rayleigh 散射截面（单位：m²）

    参数:
        wavelength_m: 波长（单位：米）
        n: 折射率

    返回:
        散射截面 σs（单位：平方米）
    """
    numerator = 8 * np.pi**3 * (n**2 - 1)**2
    denominator = 3 * wavelength_m**4
    return numerator / denominator
