# src/biooptics/simulation/run_small.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from ..models.layers import LayerStack
from ..mc.photon_types import PhotonRecord
from ..mc.tallies import EnergyTally
from ..mc.kernels_cpu import (
    _sample_hg_cos_theta, _sample_phi, _rotate_direction,
    handle_boundary,
)

@dataclass
class SmallSimConfig:
    n_photons: int = 1000
    rng_seed: int = 1234
    rr_threshold: float = 1e-3
    rr_p: float = 0.1
    max_steps: int = 10_000

def _sample_free_path(rng, mu_t: float) -> float:
    # 指数分布：-ln(U)/mu_t（保护极小 U）
    return -math.log(max(1e-12, float(rng.random()))) / mu_t

def run_small(layer_stack: LayerStack, cfg: SmallSimConfig):
    """
    返回: (R_d, T_d, A_layers)
    - R_d: 顶面出射能量（含 Fresnel 决策）
    - T_d: 底面出射能量
    - A_layers: 每层吸收能量
    """
    rng = np.random.default_rng(cfg.rng_seed)
    n_layers = len(layer_stack.layers)
    energy = EnergyTally(n_layers=n_layers)

    for _ in range(cfg.n_photons):
        # 初始化一个光子：位于顶面下方极近处，向下
        p = np.zeros((), dtype=PhotonRecord)
        p["x"] = 0.0; p["y"] = 0.0; p["z"] = np.float32(1e-7)
        p["ux"] = 0.0; p["uy"] = 0.0; p["uz"] = 1.0
        p["w"] = 1.0; p["alive"] = 1
        p["layer_idx"] = 0

        steps = 0
        while p["alive"] and steps < cfg.max_steps:
            steps += 1
            idx = int(p["layer_idx"])
            layer = layer_stack.layers[idx]
            mu_a, mu_s, g = float(layer.mu_a), float(layer.mu_s), float(layer.g)
            # src/biooptics/simulation/run_small.py 中 while 循环里，计算 mu_t 后：
            mu_t = mu_a + mu_s
            if mu_t <= 0.0:
                # 透明层特殊处理：如果是“最后一层 & 半无限 & 朝下”，认为直接逃逸到底部
                is_last = (idx == len(layer_stack.layers) - 1)
                if is_last and np.isinf(layer.d) and float(p["uz"]) > 0.0:
                    energy.T_d += float(p["w"])
                    p["alive"] = 0
                    break
                else:
                    s_intrinsic = math.inf
            else:
                s_intrinsic = _sample_free_path(rng, mu_t)


            # 查到边界的距离
            s_boundary = layer_stack.get_boundary_distance(float(p["z"]), float(p["uz"]), idx)

            # 谁先到？
            hit_boundary_first = (s_boundary < s_intrinsic)

            if hit_boundary_first:
                # 先到边界：前进到界面
                p["x"] += p["ux"] * s_boundary
                p["y"] += p["uy"] * s_boundary
                p["z"] += p["uz"] * s_boundary

                outcome = handle_boundary(p, layer_stack, rng, energy)
                if outcome == "exit_top":
                    # 已在 handle_boundary 里加过 energy.Rd
                    break
                elif outcome == "exit_bottom":
                    # 已在 handle_boundary 里加过 energy.Td
                    break
                else:
                    # 反射或层内透射，继续循环
                    continue

            # 否则：先发生体相互作用（前进到碰撞点）
            s = s_intrinsic if np.isfinite(s_intrinsic) else 0.0
            p["x"] += p["ux"] * s
            p["y"] += p["uy"] * s
            p["z"] += p["uz"] * s

            # 吸收
            if mu_t > 0.0:
                absorb = float(p["w"]) * (mu_a / mu_t)
                p["w"] = float(p["w"]) - absorb
                energy.A_layers[idx] += absorb

            # 俄罗斯轮盘
            if float(p["w"]) < cfg.rr_threshold:
                if float(rng.random()) < cfg.rr_p:
                    p["w"] = float(p["w"]) / cfg.rr_p
                else:
                    p["alive"] = 0
                    break

            # 散射（若 mu_s>0）
            if mu_s > 0.0:
                u1, u2 = float(rng.random()), float(rng.random())
                cos_t = _sample_hg_cos_theta(g, u1)
                phi = _sample_phi(u2)
                ux, uy, uz = _rotate_direction(float(p["ux"]), float(p["uy"]), float(p["uz"]), cos_t, phi)
                p["ux"], p["uy"], p["uz"] = ux, uy, uz

        # 光子生命终结：继续下一个

    # 归一化
    scale = 1.0 / float(cfg.n_photons)
    return energy.R_d * scale, energy.T_d * scale, energy.A_layers * scale
