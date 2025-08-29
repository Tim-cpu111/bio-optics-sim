import numpy as np
import math
from typing import Optional

from ..models.materials import TissueOpticalProps
from .photon_types import PhotonRecord
from .tallies import Tallies   # ← 新增：可选统计容器


def simulate_absorption_only(
    N: int,
    props: TissueOpticalProps,
    seed: int = 42,
    tallies: Optional[Tallies] = None,   # ← 新增：可选打点
):
    """
    半无限均匀介质 - 仅考虑吸收 (不散射, 不边界).
    验证目标: 所有光子最终都被吸收 => A ≈ 1.
    备注：若传入 tallies，将记录步长和轨迹点位。
    """
    rng = np.random.default_rng(seed)
    mu_a, mu_s, mu_t = props.mu_a, props.mu_s, props.mu_t

    # 初始化光子
    photons = np.zeros(N, dtype=PhotonRecord)
    photons["z"] = 1e-7  # 刚进入组织
    photons["uz"] = 1.0  # 向下
    photons["w"] = 1.0
    photons["alive"] = 1

    A_total = 0.0

    for i in range(N):
        if tallies is not None:
            tallies.start_track()

        # 解包到局部变量（便于记录）
        x = float(photons[i]["x"])
        y = float(photons[i]["y"])
        z = float(photons[i]["z"])
        uz = float(photons[i]["uz"])
        w = float(photons[i]["w"])

        if tallies is not None:
            tallies.log_pos(x, z)

        while photons[i]["alive"]:
            # 1) 随机步长
            s = -np.log(max(1e-12, float(rng.random()))) / mu_t
            x += 0.0 * s  # 无散射，方向固定向下；x,y不变
            y += 0.0 * s
            z += uz * s

            if tallies is not None:
                tallies.log_step(s)
                tallies.log_pos(x, z)

            # 2) 吸收
            absorb = w * (mu_a / mu_t)
            A_total += absorb
            w -= absorb
            if tallies is not None:
                tallies.add_absorb(absorb)

            # 3) 终止条件 (光子权重趋近于0)
            if w < 1e-6:
                photons[i]["alive"] = 0

        # 回写
        photons[i]["x"], photons[i]["y"], photons[i]["z"] = x, y, z
        photons[i]["w"] = w
        if tallies is not None:
            tallies.bump()

    # 所有光子最终都被吸收 => A_total / N ≈ 1
    return 0.0, A_total / N   # 返回 R_d, A


# --- HG 采样与方向旋转 ---

def _sample_hg_cos_theta(g: float, u: float) -> float:
    """采样 HG 相函数的 cos(theta)。u~U(0,1)"""
    if abs(g) < 1e-6:
        return 1.0 - 2.0 * u  # 各向同性特例
    num = 1.0 - g*g
    den = 1.0 - g + 2.0*g*u
    cos_theta = (1.0 + g*g - (num/den)**2) / (2.0*g)
    # 数值安全
    return max(-1.0, min(1.0, float(cos_theta)))

def _sample_phi(u: float) -> float:
    return 2.0 * math.pi * u

def _rotate_direction(ux, uy, uz, cos_t, phi):
    """把(ux,uy,uz)按(theta,phi)旋转到新方向，返回(ux',uy',uz')。
       处理 |uz|≈1 的数值稳定分支。
    """
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t*cos_t))
    if abs(uz) > 0.99999:
        # 接近极轴，选择局部正交基直接旋转
        ux_p = sin_t * math.cos(phi)
        uy_p = sin_t * math.sin(phi)
        uz_p = math.copysign(1.0, uz) * cos_t
        return ux_p, uy_p, uz_p

    # 构造与 (ux,uy,uz) 正交的单位向量 (vx,vy,vz)
    denom = math.sqrt(1.0 - uz*uz)
    vx = -uy / denom
    vy =  ux / denom
    vz =  0.0

    # 另一个正交向量 (wx,wy,wz) = v × u
    wx = vy*uz - vz*uy
    wy = vz*ux - vx*uz
    wz = vx*uy - vy*ux

    # 在 (u,v,w) 的局部坐标中旋转：u' = cos_t*u + sin_t*(cosφ*v + sinφ*w)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    ux_p = cos_t*ux + sin_t*(cos_p*vx + sin_p*wx)
    uy_p = cos_t*uy + sin_t*(cos_p*vy + sin_p*wy)
    uz_p = cos_t*uz + sin_t*(cos_p*vz + sin_p*wz)

    # 归一化抑制漂移
    norm = math.sqrt(ux_p*ux_p + uy_p*uy_p + uz_p*uz_p) or 1.0
    return ux_p/norm, uy_p/norm, uz_p/norm


def simulate_with_scattering(
    N: int,
    props: TissueOpticalProps,
    seed: int = 42,
    rr_threshold: float = 1e-3,   # 俄罗斯轮盘阈值
    rr_p: float = 0.1,            # 存活概率
    max_steps: int = 10000,       # 安全上限，避免极端参数时长循环
    tallies: Optional[Tallies] = None,  # ← 新增：可选打点
):
    """
    半无限均匀介质（仍不处理边界），加入 HG 散射与吸收。
    输出: (R_d, A)，此阶段 R_d 固定为 0（没有边界出射），A ≈ 1。
    备注：若传入 tallies，将记录步长、方向 uz 与轨迹。
    """
    rng = np.random.default_rng(seed)
    mu_a, mu_s, mu_t, g = props.mu_a, props.mu_s, props.mu_t, props.g

    photons = np.zeros(N, dtype=PhotonRecord)
    photons["z"] = 1e-7
    photons["uz"] = 1.0
    photons["w"] = 1.0
    photons["alive"] = 1

    A_total = 0.0

    for i in range(N):
        if tallies is not None:
            tallies.start_track()

        # 解包到局部变量
        x  = float(photons[i]["x"]);  y  = float(photons[i]["y"]);  z  = float(photons[i]["z"])
        ux = float(photons[i]["ux"]); uy = float(photons[i]["uy"]); uz = float(photons[i]["uz"])
        w  = float(photons[i]["w"])

        if tallies is not None:
            tallies.log_pos(x, z)

        steps = 0
        while photons[i]["alive"] and steps < max_steps:
            steps += 1
            # 1) 走步长
            s = -math.log(max(1e-12, float(rng.random()))) / mu_t
            x += ux*s; y += uy*s; z += uz*s

            if tallies is not None:
                tallies.log_step(s)
                tallies.log_pos(x, z)

            # 2) 吸收
            absorb = w * (mu_a / mu_t)
            w -= absorb
            A_total += absorb
            if tallies is not None:
                tallies.add_absorb(absorb)

            # 3) 俄罗斯轮盘（加速收敛）
            if w < rr_threshold:
                if rng.random() < rr_p:
                    w /= rr_p
                else:
                    photons[i]["alive"] = 0
                    break

            # 4) 散射：HG
            u1 = float(rng.random()); u2 = float(rng.random())
            cos_t = _sample_hg_cos_theta(g, u1)
            phi   = _sample_phi(u2)
            ux, uy, uz = _rotate_direction(ux, uy, uz, cos_t, phi)

            if tallies is not None:
                tallies.log_uz(uz)

        # 写回与结束
        photons[i]["x"], photons[i]["y"], photons[i]["z"] = x, y, z
        photons[i]["ux"], photons[i]["uy"], photons[i]["uz"] = ux, uy, uz
        photons[i]["w"] = w
        photons[i]["alive"] = 0
        if tallies is not None:
            tallies.bump()

    R_d = 0.0  # 还未模拟出射
    A   = A_total / N
    return R_d, A
