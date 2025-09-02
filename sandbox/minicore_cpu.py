
import numpy as np

class Photon:
    def __init__(self, pos, dir, w=1.0):
        self.pos = np.array(pos, dtype=float) #[x, y, z]
        self.dir = np.array(dir, dtype=float) #[ux, uy, uz]
        self.w = w
        self.alive = True

"""采样步长: s = -ln(ξ)/μ_t"""
def step_length(mu_t, rng = np.random):
    if mu_t <= 0:
        raise ValueError("mu_t must be >0")
    xi = max(rng.rand(), 1e-6)
    return -np.log(xi)/mu_t


"""吸收权重更新: Δw = w * (μa/μt)"""
def absorb(photon, mu_a, mu_t):
    dw = photon.w *(mu_a/mu_t)
    photon.w -= dw
    return dw


"""各向同性散射方向更新"""
def scatter_isotropic(photon, rng=np.random):
    
    cos_theta = 2 * rng.rand() - 1
    phi = 2 * np.pi * rng.rand()
    sin_theta = np.sprt(1 - cos_theta**2)
    photon.dir = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])