import numpy as np
import math
from typing import Tuple, Optional, Literal

from ..models.materials import TissueOpticalProps
from .photon_types import PhotonRecord
from .tallies import Tallies   # ← 新增：可选统计容器

# --- 新增：无偏振平均的 Fresnel 反射率（从组织→空气边界） ---
def _fresnel_R_unpolarized(cos_i: float, n1: float, n2: float) -> float:
    """
    返回无偏振平均 Fresnel 反射率 R。cos_i 是入射角与法向的余弦（取 0..1）。
    n1: 介质内折射率（组织），n2: 外侧折射率（空气=1.0）
    处理全反射：若发生 TIR，R=1。
    """
    cos_i = max(0.0, min(1.0, float(cos_i)))
    # Snell
    n_ratio = n1 / n2
    sin_t2 = (n_ratio * n_ratio) * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return 1.0  # 全反射
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t2))
    # s、p 两个偏振分量
    rs_num = n1 * cos_i - n2 * cos_t
    rs_den = n1 * cos_i + n2 * cos_t
    rp_num = n1 * cos_t - n2 * cos_i
    rp_den = n1 * cos_t + n2 * cos_i
    Rs = (rs_num / rs_den) ** 2
    Rp = (rp_num / rp_den) ** 2
    return 0.5 * (Rs + Rp)

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
    rr_threshold: float = 1e-3,
    rr_p: float = 0.1,
    max_steps: int = 10000,
    tallies: "Optional[Tallies]" = None,   # 保持之前的可选打点
    progress: bool = False,                 # 若你已集成 tqdm，可保留；否则忽略
    with_boundary: bool = True,   # ← 新增：是否启用上表面边界 + Fresnel
):
    """
    半无限均匀介质（z>0 为组织，z=0 是上表面；空气在 z<0）。
    本版本：吸收 + 散射 + 上表面 Fresnel 出射（统计 R_d）。
    """
    # 进度条优雅降级（可选）
    try:
        from tqdm import trange
    except Exception:
        trange = range

    rng = np.random.default_rng(seed)
    mu_a, mu_s, mu_t, g, n_in, n_out = props.mu_a, props.mu_s, props.mu_t, props.g, props.n, 1.0

    photons = np.zeros(N, dtype=PhotonRecord)
    photons["z"] = 1e-7
    photons["uz"] = 1.0
    photons["w"] = 1.0
    photons["alive"] = 1

    A_total, Rd_total = 0.0, 0.0

    for i in trange(N, disable=not progress, desc="MC Scattering+Fresnel"):
        if tallies is not None:
            tallies.start_track()

        # 解包到局部变量
        x  = float(photons[i]["x"]);  y  = float(photons[i]["y"]);  z  = float(photons[i]["z"])
        ux = float(photons[i]["ux"]); uy = float(photons[i]["uy"]); uz = float(photons[i]["uz"])
        w  = float(photons[i]["w"])

        if tallies is not None:
            tallies.log_pos(x, z)

        steps = 0
        alive = True
        while alive and steps < max_steps:
            steps += 1
    # 采样一次步长
            s = -math.log(max(1e-12, float(rng.random()))) / mu_t

            if with_boundary:
                s_rem = s
                while True:
                    s_to_boundary = (z / (-uz)) if uz < 0.0 else math.inf
                    if s_to_boundary <= s_rem:
                        # 到边界
                        x += ux * s_to_boundary
                        y += uy * s_to_boundary
                        z += uz * s_to_boundary
                        if tallies is not None:
                            tallies.log_step(s_to_boundary)
                            tallies.log_pos(x, z)

                        # Fresnel
                        cos_i = abs(uz)
                        R = _fresnel_R_unpolarized(cos_i, n_in, n_out)
                        if rng.random() > R:
                            # 透射出射
                            Rd_total += w * (1.0 - R)
                            if tallies is not None:
                                tallies.add_reflect(w * (1.0 - R))
                                tallies.bump()
                            alive = False
                            break
                        else:
                            # 反射回介质
                            uz = -uz
                            z = 1e-7
                            s_rem -= s_to_boundary
                            if s_rem > 0.0:
                                continue
                            else:
                                break
                    else:
                        # 不撞边界，走完余量
                        x += ux * s_rem
                        y += uy * s_rem
                        z += uz * s_rem
                        if tallies is not None:
                            tallies.log_step(s_rem)
                            tallies.log_pos(x, z)
                        break

                    if not alive:
                        break
            else:
                # 无边界模式：直接走完整个步长
                x += ux * s
                y += uy * s
                z += uz * s
                if tallies is not None:
                    tallies.log_step(s)
                    tallies.log_pos(x, z)
            
            # 1) 采样一次“碰撞步长”（到达散射/吸收中心的距离）
            # s = -math.log(max(1e-12, float(rng.random()))) / mu_t
            # s_rem = s

            # # 1a) 在走这段距离前，检查是否会撞到上表面（z=0）
            # while True:
            #     if uz < 0.0:
            #         # 距离上表面（沿当前方向）
            #         s_to_boundary = z / (-uz)  # z>0, uz<0 时为正
            #     else:
            #         s_to_boundary = math.inf

            #     if s_to_boundary <= s_rem:
            #         # 会先到边界：先走到 z=0
            #         x += ux * s_to_boundary
            #         y += uy * s_to_boundary
            #         z += uz * s_to_boundary
            #         if tallies is not None:
            #             tallies.log_step(s_to_boundary)
            #             tallies.log_pos(x, z)

            #         # 在边界做 Fresnel 判定
            #         cos_i = abs(uz)  # 与法向的夹角余弦
            #         R = _fresnel_R_unpolarized(cos_i, n_in, n_out)

            #         if rng.random() > R:
            #             # 透射到空气：计入漫反射，光子死亡
            #             Rd_total += w * (1.0 - R)
            #             if tallies is not None:
            #                 tallies.add_reflect(w * (1.0 - R))
            #                 tallies.bump()
            #             alive = False
            #             break
            #         else:
            #             # 反射回组织：法向分量反号（镜面反射），位置贴边界内侧
            #             uz = -uz
            #             z = 1e-7  # 贴回组织内部，避免数值抖动
            #             # 剩余的步长继续走（不产生吸收/散射，直到真正到达碰撞点）
            #             s_rem -= s_to_boundary
            #             if s_rem > 0.0:
            #                 # 继续检查是否又撞边界（很斜时可能多次反射）
            #                 continue
            #             else:
            #                 # 没有剩余步长，说明此“碰撞步长”恰好被边界耗尽
            #                 # 不发生吸收/散射，进入下一轮重新采样新的步长
            #                 break
            #     else:
            #         # 不会撞边界：直接走完这段 s_rem
            #         x += ux * s_rem
            #         y += uy * s_rem
            #         z += uz * s_rem
            #         if tallies is not None:
            #             tallies.log_step(s_rem)
            #             tallies.log_pos(x, z)
            #         # 结束“是否撞边界”的内循环，进入在碰撞点的吸收/散射
            #         break

            # if not alive:
            #     break  # 光子已出射

            # 2) 到达碰撞点：按 μa/μt 扣减权重（吸收），其余用于散射
            absorb = w * (mu_a / mu_t)
            w -= absorb
            A_total += absorb
            if tallies is not None:
                tallies.add_absorb(absorb)

            # 3) 俄罗斯轮盘（避免极小权重长尾）
            if w < rr_threshold:
                if rng.random() < rr_p:
                    w /= rr_p
                else:
                    if tallies is not None:
                        tallies.bump()
                    alive = False
                    break

            # 4) 散射：HG 采样 + 方向旋转
            u1 = float(rng.random()); u2 = float(rng.random())
            cos_t = _sample_hg_cos_theta(g, u1)
            phi   = _sample_phi(u2)
            ux, uy, uz = _rotate_direction(ux, uy, uz, cos_t, phi)
            if tallies is not None:
                tallies.log_uz(uz)

        # 写回与死亡标记
        photons[i]["x"], photons[i]["y"], photons[i]["z"] = x, y, z
        photons[i]["ux"], photons[i]["uy"], photons[i]["uz"] = ux, uy, uz
        photons[i]["w"] = w
        photons[i]["alive"] = 0

    R_d = Rd_total / N
    A   = A_total / N
    return R_d, A

def snell_refract(
    u_incident: np.ndarray,
    n1: float,
    n2: float
) -> Tuple[bool, Optional[np.ndarray], float, Optional[float]]:
    """
    Snell + TIR 判定 + 折射方向向量
    约定：层法向取 +z，函数内部按“使 cos(theta_i)>0”的法向进行几何计算。
    返回:
      - tir: 是否全反射
      - u_trans: 折射方向（单位向量）；TIR 时为 None
      - cos_i: 入射角余弦 (= |u_z|)
      - cos_t: 折射角余弦；TIR 时为 None
    """
    u = np.asarray(u_incident, dtype=float)
    # 单位化，避免外部传入的向量有误差
    norm = np.linalg.norm(u)
    if norm == 0.0:
        raise ValueError("u_incident must be a non-zero vector")
    u = u / norm

    uz = u[2]
    # 选择法向 N，使得 cos(theta_i) = -N·u = |uz| > 0
    # 若 uz>0（朝下撞下界），取 N = -z_hat；若 uz<0（朝上撞上界），取 N = +z_hat
    N = np.array([0.0, 0.0, -1.0 if uz > 0.0 else 1.0], dtype=float)
    cos_i = -np.dot(N, u)  # = |uz|

    # 折射率比
    eta = n1 / n2
    sin2_t = eta**2 * (1.0 - cos_i**2)

    # 全反射判据
    if sin2_t > 1.0:
        return True, None, float(cos_i), None

    # 计算 cos(theta_t)
    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))

    # 计算折射方向向量 (vector form of Snell)
    # t = eta * u + (eta*cos_i - cos_t) * N
    t = eta * u + (eta * cos_i - cos_t) * N

    # 数值稳健：单位化
    t_norm = np.linalg.norm(t)
    if t_norm == 0.0:
        # 理论上不应出现，这里作为保护
        return True, None, float(cos_i), None
    t = t / t_norm

    return False, t, float(cos_i), float(cos_t)


def fresnel_unpolarized(n1: float, n2: float, cos_i: float, cos_t: float) -> float:
    """
    无偏振 Fresnel 反射率
    输入: n1, n2 - 入射/透射介质折射率
         cos_i  - 入射角余弦 (>=0)
         cos_t  - 折射角余弦 (>=0)
    返回: R in [0,1]
    """
    # 避免除零
    denom_s = (n1 * cos_i + n2 * cos_t)
    denom_p = (n1 * cos_t + n2 * cos_i)

    if denom_s == 0.0 or denom_p == 0.0:
        return 1.0  # 极限情况，全反射

    Rs = ((n1 * cos_i - n2 * cos_t) / denom_s) ** 2
    Rp = ((n1 * cos_t - n2 * cos_i) / denom_p) ** 2
    R = 0.5 * (Rs + Rp)

    # 数值稳健：限制在 [0,1]
    return float(min(max(R, 0.0), 1.0))


def handle_boundary(photon, stack, rng) -> Literal["reflected","transmitted","exit_top","exit_bottom"]:
    raise ImportError
