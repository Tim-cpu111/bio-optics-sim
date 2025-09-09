from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_with_scattering

def test_energy_balance_with_boundary():
    props = TissueOpticalProps(mu_a=0.1, mu_s=10.0, g=0.9, n=1.37)
    R_d, A = simulate_with_scattering(N=50000, props=props, seed=42, with_boundary=True)
    assert abs(R_d + A - 1.0) < 0.02   # 统计误差随 N 收敛
    assert R_d > 0.02                  # 有明显出射

