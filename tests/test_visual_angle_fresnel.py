# tests/test_visual_angle_fresnel.py
import numpy as np
import pytest

from biooptics.models.layers import Layer, LayerStack
from biooptics.mc.photon_types import PhotonRecord
from biooptics.mc.kernels_cpu import snell_refract, fresnel_unpolarized, handle_boundary

class RNGDet:
    """可复用的 RNG，包装 default_rng。"""
    def __init__(self, seed=0): self._rng = np.random.default_rng(seed)
    def random(self): return float(self._rng.random())

def _trial_exit_top_probability(theta_deg: float, n1=1.37, n2=1.0, trials=5000, seed=0):
    # 单层：纯透明、半无限
    stack = LayerStack([Layer(mu_a=0.0, mu_s=0.0, g=0.0, n=n1, d=np.inf)])
    rng = RNGDet(seed)

    Rd_sum = 0.0
    for _ in range(trials):
        # 放在顶界，方向朝上（组织→空气）
        theta = np.deg2rad(theta_deg)
        uz = -float(np.cos(theta))   # 朝上 → uz<0
        ux =  float(np.sin(theta));  uy = 0.0

        p = np.zeros((), dtype=PhotonRecord)
        p["x"]=0.0; p["y"]=0.0; p["z"]=0.0
        p["ux"]=ux; p["uy"]=uy; p["uz"]=uz
        p["w"]=1.0; p["alive"]=1; p["layer_idx"]=0

        # 简单 tally（只要 R_d/T_d 字段）
        tallies = type("T", (), {"R_d":0.0, "T_d":0.0})()
        outcome = handle_boundary(p, stack, rng, tallies)

        # 顶出射由 handle_boundary 直接累计到 tallies.R_d
        Rd_sum += tallies.R_d

    Rd_mc = Rd_sum / trials  # MC 估计的 顶出射概率

    # 理论值：内部反射率 R_internal（组织→空气）
    # 先用 snell 拿 cos_i/cos_t（用 incident 向量的 |uz| 即可）
    u_inc = np.array([ux, uy, -uz], dtype=float)  # 方向只需 |uz| 用处，这里随便给同角度
    tir, _, cos_i, cos_t = snell_refract(u_incident=np.array([ux,0.0,abs(uz)]), n1=n1, n2=n2)
    if tir:
        R_int = 1.0
    else:
        R_int = fresnel_unpolarized(n1, n2, cos_i, cos_t)

    Rd_theory = 1.0 - R_int  # 顶出射概率 = 1 - 内部反射

    return Rd_mc, Rd_theory

@pytest.mark.parametrize("deg", [0, 15, 30, 45])
def test_top_exit_matches_internal_fresnel(deg):
    Rd_mc, Rd_th = _trial_exit_top_probability(theta_deg=deg, trials=4000, seed=2025)
    # 统计误差容差放宽一点
    assert Rd_mc == pytest.approx(Rd_th, rel=0.0, abs=0.02)
