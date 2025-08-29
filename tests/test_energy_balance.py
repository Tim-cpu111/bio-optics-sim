from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_absorption_only

def test_absorption_only():
    props = TissueOpticalProps(mu_a=1.0, mu_s=0.0, g=0.0, n=1.37)
    R_d, A = simulate_absorption_only(N=1000, props=props, seed=123)
    # 所有光子都被吸收 => A ≈ 1
    assert abs(A - 1.0) < 0.05
    assert abs(R_d) < 1e-6

from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_with_scattering

def test_energy_balance_with_scattering():
    # 既有散射又有吸收（不考虑边界），最终应 A≈1, R_d≈0
    props = TissueOpticalProps(mu_a=0.2, mu_s=5.0, g=0.9, n=1.37)
    R_d, A = simulate_with_scattering(N=20000, props=props, seed=123)
    assert abs(R_d) < 1e-6
    assert abs(A - 1.0) < 0.03   # 允许 3% 统计误差（可按N调整）
