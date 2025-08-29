import numpy as np
from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import _sample_hg_cos_theta

def test_hg_isotropic_g0_mean_cos_zero():
    rng = np.random.default_rng(0)
    g = 0.0
    M = 100000
    cos_t = np.array([_sample_hg_cos_theta(g, float(rng.random())) for _ in range(M)])
    # 各向同性: E[cosθ] ≈ 0
    assert abs(cos_t.mean()) < 0.01
