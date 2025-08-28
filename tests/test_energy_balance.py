from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_absorption_only

def test_absorption_only():
    props = TissueOpticalProps(mu_a=1.0, mu_s=0.0, g=0.0, n=1.37)
    R_d, A = simulate_absorption_only(N=1000, props=props, seed=123)
    # 所有光子都被吸收 => A ≈ 1
    assert abs(A - 1.0) < 0.05
    assert abs(R_d) < 1e-6
