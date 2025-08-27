import sys
import os

# 把 src 加入模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from biooptics.models.scattering import rayleigh_scattering_cross_section


def test_rayleigh_scattering_cross_section():
    wavelength = 500e-9  # 500 nm
    n = 1.4
    result = rayleigh_scattering_cross_section(wavelength, n)
    assert result > 0
    assert isinstance(result, float)


