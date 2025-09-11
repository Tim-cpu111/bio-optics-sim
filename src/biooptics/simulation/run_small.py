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
    """指数分布步长采样"""
    return -math.log(max(1e-12, float(rng.random()))) / mu_t

def run_small(layer_stack: LayerStack,
              cfg: SmallSimConfig,
              viz=None,
              viz_first_photons: int = 50,
              viz_step_stride: int = 3,
              viz_max_points: int = 20000):
    """
    多层蒙特卡罗小规模仿真：
    返回 (R_d, T_d, A_layers)
    - R_d: 顶出射能量
    - T_d: 底出射能量
    - A_layers: 每层吸收能量数组

    可选 viz: 提供 log_step(s), log_uz(uz), log_pos(x,z) 接口的对象，用于采样可视化。
    """
    rng = np.random.default_rng(cfg.rng_seed)
    n_layers = len(layer_stack.layers)
    energy = EnergyTally(n_layers=n_layers)

    total_points = 0

    for p_idx in range(cfg.n_photons):
        # 初始化光子：顶面下方极近处，向下
        p = np.zeros((), dtype=PhotonRecord)
        p["x"], p["y"], p["z"] = 0.0, 0.0, np.float32(1e-7)
        p["ux"], p["uy"], p["uz"] = 0.0, 0.0, 1.0
        p["w"], p["alive"], p["layer_idx"] = 1.0, 1, 0

        steps = 0
        while p["alive"] and steps < cfg.max_steps:
            steps += 1
            idx = int(p["layer_idx"])
            layer = layer_stack.layers[idx]
            mu_a, mu_s, g = float(layer.mu_a), float(layer.mu_s), float(layer.g)
            mu_t = mu_a + mu_s

            # 特殊情况：透明半无限层 → 直接逃逸
            if mu_t <= 0.0:
                is_last = (idx == n_layers - 1)
                if is_last and np.isinf(layer.d) and float(p["uz"]) > 0.0:
                    energy.T_d += float(p["w"])
                    p["alive"] = 0
                    break
                s_intrinsic = math.inf
            else:
                s_intrinsic = _sample_free_path(rng, mu_t)

            # 到边界的距离
            s_boundary = layer_stack.get_boundary_distance(float(p["z"]), float(p["uz"]), idx)

            # 边界优先
            if s_boundary < s_intrinsic:
                p["x"] += p["ux"] * s_boundary
                p["y"] += p["uy"] * s_boundary
                p["z"] += p["uz"] * s_boundary
                if viz and p_idx < viz_first_photons and steps % viz_step_stride == 0 and total_points < viz_max_points:
                    viz.log_step(float(s_boundary)); viz.log_pos(float(p["x"]), float(p["z"]))
                    total_points += 1

                outcome = handle_boundary(p, layer_stack, rng, energy)
                if outcome in ("exit_top", "exit_bottom"):
                    break
                else:
                    continue

            # 否则体相互作用
            s = 0.0 if not np.isfinite(s_intrinsic) else float(s_intrinsic)
            p["x"] += p["ux"] * s
            p["y"] += p["uy"] * s
            p["z"] += p["uz"] * s
            if viz and p_idx < viz_first_photons and steps % viz_step_stride == 0 and total_points < viz_max_points:
                viz.log_step(s); viz.log_pos(float(p["x"]), float(p["z"]))
                total_points += 1

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

            # 散射
            if mu_s > 0.0:
                u1, u2 = float(rng.random()), float(rng.random())
                cos_t = _sample_hg_cos_theta(g, u1)
                phi   = _sample_phi(u2)
                ux, uy, uz = _rotate_direction(float(p["ux"]), float(p["uy"]), float(p["uz"]), cos_t, phi)
                p["ux"], p["uy"], p["uz"] = ux, uy, uz
                if viz and p_idx < viz_first_photons and steps % viz_step_stride == 0 and total_points < viz_max_points:
                    viz.log_uz(float(p["uz"]))
                    total_points += 1

    # 归一化
    scale = 1.0 / float(cfg.n_photons)
    return energy.R_d * scale, energy.T_d * scale, energy.A_layers * scale
