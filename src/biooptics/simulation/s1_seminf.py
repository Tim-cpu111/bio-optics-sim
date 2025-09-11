# src/biooptics/simulation/s1_seminf.py
from __future__ import annotations
import math
import numpy as np

from ..models.materials import TissueOpticalProps
from ..mc.tallies import Tallies  # 如果需要打点
from ..mc.kernels_cpu import (
    _sample_hg_cos_theta, _sample_phi, _rotate_direction,
    _fresnel_R_unpolarized,  # 你已有的无偏振 R(cos_i,n1,n2)
)

def simulate_with_scattering(
    N: int,
    props: TissueOpticalProps,
    seed: int = 42,
    rr_threshold: float = 1e-3,
    rr_p: float = 0.1,
    max_steps: int = 10000,
    tallies: "Tallies|None" = None,
    progress: bool = False,
    with_boundary: bool = True,
):
    """
    半无限单层（z>0 为介质，z=0 顶面，z<0 空气）。
    返回 (R_d, A)；不含 T_d（半无限假设下 = 0）。
    """
    try:
        from tqdm import trange
    except Exception:
        trange = range

    rng = np.random.default_rng(seed)
    mu_a, mu_s, g, n_in, n_out = props.mu_a, props.mu_s, props.g, props.n, 1.0
    mu_t = mu_a + mu_s

    photons = np.zeros(N, dtype=[
        ("x","f4"),("y","f4"),("z","f4"),
        ("ux","f4"),("uy","f4"),("uz","f4"),
        ("w","f4"),("alive","i1")
    ])
    photons["z"] = 1e-7
    photons["uz"] = 1.0
    photons["w"]  = 1.0
    photons["alive"] = 1

    A_total, Rd_total = 0.0, 0.0

    for i in trange(N, disable=not progress, desc="S1 MC"):
        if tallies is not None: tallies.start_track()

        x,y,z = float(photons[i]["x"]), float(photons[i]["y"]), float(photons[i]["z"])
        ux,uy,uz = float(photons[i]["ux"]), float(photons[i]["uy"]), float(photons[i]["uz"])
        w = float(photons[i]["w"])

        steps = 0
        alive = True
        while alive and steps < max_steps:
            steps += 1
            # 1) 采样碰撞步长
            s = -math.log(max(1e-12, float(rng.random()))) / max(mu_t, 1e-12)

            if with_boundary:
                s_rem = s
                while True:
                    s_to_boundary = (z / (-uz)) if uz < 0.0 else math.inf
                    if s_to_boundary <= s_rem:
                        # 到边界
                        x += ux * s_to_boundary; y += uy * s_to_boundary; z += uz * s_to_boundary
                        if tallies is not None:
                            tallies.log_step(s_to_boundary); tallies.log_pos(x, z)
                        # Fresnel 透/反
                        cos_i = abs(uz)
                        R = _fresnel_R_unpolarized(cos_i, n_in, n_out)
                        if rng.random() > R:
                            # 透射出射
                            Rd_total += w * (1.0 - R)
                            if tallies is not None: tallies.add_reflect(w * (1.0 - R)); tallies.bump()
                            alive = False
                            break
                        else:
                            # 反射回介质
                            uz = -uz
                            z = 1e-7
                            s_rem -= s_to_boundary
                            if s_rem > 0.0: continue
                            else: break
                    else:
                        x += ux * s_rem; y += uy * s_rem; z += uz * s_rem
                        if tallies is not None:
                            tallies.log_step(s_rem); tallies.log_pos(x, z)
                        break
                if not alive: break
            else:
                x += ux * s; y += uy * s; z += uz * s
                if tallies is not None:
                    tallies.log_step(s); tallies.log_pos(x, z)

            # 2) 吸收
            absorb = w * (mu_a / max(mu_t, 1e-12))
            w -= absorb; A_total += absorb
            if tallies is not None: tallies.add_absorb(absorb)

            # 3) 俄罗斯轮盘
            if w < rr_threshold:
                if rng.random() < rr_p:
                    w /= rr_p
                else:
                    if tallies is not None: tallies.bump()
                    alive = False
                    break

            # 4) 散射
            if mu_s > 0.0:
                u1, u2 = float(rng.random()), float(rng.random())
                cos_t = _sample_hg_cos_theta(g, u1)
                phi   = _sample_phi(u2)
                ux, uy, uz = _rotate_direction(ux, uy, uz, cos_t, phi)

        photons[i]["x"], photons[i]["y"], photons[i]["z"] = x,y,z
        photons[i]["ux"], photons[i]["uy"], photons[i]["uz"] = ux,uy,uz
        photons[i]["w"] = w
        photons[i]["alive"] = 0

    return Rd_total / N, A_total / N
