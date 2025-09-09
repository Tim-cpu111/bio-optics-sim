from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_with_scattering

def test_energy_balance_with_scattering():
    # 不考虑边界 → 预期 A≈1, R_d≈0
    props = TissueOpticalProps(mu_a=0.2, mu_s=5.0, g=0.9, n=1.37)
    R_d, A = simulate_with_scattering(N=20000, props=props, seed=123, with_boundary=False)
    assert abs(R_d) < 1e-6
    assert abs(A - 1.0) < 0.03
