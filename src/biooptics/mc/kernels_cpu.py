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




def snell_refract(
    u_incident: np.ndarray,
    n1: float,
    n2: float
) -> Tuple[bool, Optional[np.ndarray], float, Optional[float]]:
    """
    计算光子在界面折射/全反射的方向。

    参数:
        u_incident : ndarray, 入射方向 (单位向量)
        n1, n2     : float, 入射/透射介质折射率

    返回:
        tir   : bool, 是否发生全反射
        u_trans: ndarray|None, 折射方向向量 (单位化); TIR 时为 None
        cos_i : float, 入射角余弦 = |uz|
        cos_t : float|None, 折射角余弦; TIR 时为 None

    关键点:
    - 用 Snell 定律 n1*sinθi = n2*sinθt 判定 TIR。
    - 折射向量用向量公式计算: t = ηu + (ηcosθi - cosθt)N。
    - 返回结果保证能量几何一致，且向量单位化。
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
    计算无偏振光的 Fresnel 反射率。

    参数:
        n1, n2 : float, 入射/透射介质折射率
        cos_i  : float, 入射角余弦
        cos_t  : float, 折射角余弦

    返回:
        R : float, 反射率 (0~1)

    关键点:
    - Rs, Rp 分别为 s/p 偏振的反射率。
    - R = (Rs + Rp) / 2。
    - 数值稳健: 避免除零，结果限制在 [0,1]。
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


def handle_boundary(photon, stack, rng, tallies) -> Literal["reflected","transmitted","exit_top","exit_bottom"]:
    #初始化
    n_out = 1.0 # 初始化外界空气
    ux, uy, uz = photon["ux"], photon["uy"], photon["uz"]
    layer_idx = photon["layer_idx"]

    """1. 根据 uz 判定光子撞到上界还是下界：
       - uz > 0  → 下边界，候选层 j = i + 1
       - uz < 0  → 上边界，候选层 j = i - 1"""
    if uz >= 0:
        layer_idx_next = layer_idx + 1
    elif uz < 0:
        layer_idx_next = layer_idx - 1

    """    2. 获取当前层折射率 n1 与目标介质折射率 n2：
       - 若 j 在 [0, L-1] → n2 = stack.layers[j].n
       - 若 j < 0 → 上方外界（空气），n2 = 1.0
       - 若 j >= L → 下方外界，n2 = 1.0"""
    L = stack.len_layer()
    n1 = stack.layers[layer_idx].n
    

    if layer_idx_next >= 0 and layer_idx_next <= L-1:
        n2 = stack.layers[layer_idx_next].n
    else:
        n2 = n_out

    """
     3. 调用 snell_refract(u, n1, n2)：
       - 若返回 TIR=True → 进行镜面反射 (uz 取反)，layer_idx 不变，返回 "reflected"
       - 否则得到 cos_i, cos_t, 折射方向 u_trans"""
    u_incident = np.array([ux,uy,uz], dtype= float)
    tir, u_trans, cos_i, cos_t = snell_refract(u_incident, n1, n2)

    if tir == True:
        photon["uz"] = -uz
        norm = float(np.sqrt(photon["ux"]**2 + photon["uy"]**2 + photon["uz"]**2))
        photon["ux"] /= norm
        photon["uy"] /= norm
        photon["uz"] /= norm
        # 可选：把 z 往反射后方向推进一个很小步长
        #photon["layer_idx"] = int(layer_idx)
        photon["z"] += np.sign(photon["uz"]) * 1e-12

        return "reflected"
    
    else:
        R = fresnel_unpolarized(n1, n2, cos_i, cos_t)
        xi = rng.random()
        if xi < R:
            #反射
            photon["uz"] = -uz
            norm = float(np.sqrt(photon["ux"]**2 + photon["uy"]**2 + photon["uz"]**2))
            photon["ux"] /= norm
            photon["uy"] /= norm
            photon["uz"] /= norm
            # 可选：把 z 往反射后方向推进一个很小步长
            #photon["layer_idx"] = int(layer_idx_next)
            photon["z"] += np.sign(photon["uz"]) * 1e-12
            return "reflected"
        else:
            #透射
            ux_t, uy_t, uz_t = float(u_trans[0]), float(u_trans[1]), float(u_trans[2])
            norm_t = float(np.sqrt(ux_t**2 + uy_t**2 + uz_t**2))
            if norm_t > 0.0:
                ux_t, uy_t, uz_t = ux_t / norm_t, uy_t / norm_t, uz_t / norm_t

            # 判断目标是否是内部层（0 <= j < L）还是外界（j<0 顶部 / j>=L 底部）
            if 0 <= layer_idx_next < L:
                # ---------- 透射进入相邻“内部层” ----------
                photon["ux"] = ux_t
                photon["uy"] = uy_t
                photon["uz"] = uz_t
                photon["layer_idx"] = int(layer_idx_next)
                # 把 z 轻推入新层内部，避免下一步又被判在界面
                photon["z"] += np.sign(uz_t) * 1e-12
                return "transmitted"
            else:
                
                # ---------- 透射进入“外界” → 出射并记账 ----------
                if uz > 0:  # 原本向下撞“下边界”，越界则为底部出射
                    if tallies is not None:
                        tallies.T_d += float(photon["w"])
                    photon["alive"] = 0
                    return "exit_bottom"
                else:       # 原本向上撞“上边界”，越界则为顶部出射
                    if tallies is not None:
                        tallies.R_d += float(photon["w"])
                    photon["alive"] = 0
                    return "exit_top"

    """
    处理光子在层间界面发生的事件。
    本函数整合 Snell 定律、Fresnel 反射率与全反射 (TIR) 判定，
    决定光子在界面处是反射、透射，还是从顶层/底层出射。

    输入参数
    --------
    photon : PhotonRecord
        光子的当前状态，至少包含:
          - ux, uy, uz : 光子方向向量（近似单位向量）
          - z          : 光子当前位置 z（调用时应在界面处）
          - layer_idx  : 当前所在层索引（0 开始）
          - w          : 光子权重（本函数不拆分权重）
          - alive      : 布尔标志，表示光子是否仍在传播
          
    stack : LayerStack
        多层组织数据结构，至少提供：
          - layers[i].n   : 第 i 层的折射率
          - boundaries    : 累计厚度数组（用于判定界面）
    rng : 随机数发生器
        提供 .random() 方法，返回 [0,1) 的浮点数，用于伯努利采样。
    tallies : 可选
        统计量对象，若提供则更新：
          - tallies.R_d   : 顶面反射出射的累积能量
          - tallies.T_d   : 底面透射出射的累积能量

    返回值
    ------
    outcome : 字符串
        表示光子在界面的事件结果，可能取：
          - "reflected"   : 在界面反射（含全反射）
          - "transmitted": 成功透射到相邻层
          - "exit_top"    : 从顶面出射
          - "exit_bottom" : 从底面出射

    算法步骤
    --------
    1. 根据 uz 判定光子撞到上界还是下界：
       - uz > 0  → 下边界，候选层 j = i + 1
       - uz < 0  → 上边界，候选层 j = i - 1
    2. 获取当前层折射率 n1 与目标介质折射率 n2：
       - 若 j 在 [0, L-1] → n2 = stack.layers[j].n
       - 若 j < 0 → 上方外界（空气），n2 = 1.0
       - 若 j >= L → 下方外界，n2 = 1.0
    3. 调用 snell_refract(u, n1, n2)：
       - 若返回 TIR=True → 进行镜面反射 (uz 取反)，layer_idx 不变，返回 "reflected"
       - 否则得到 cos_i, cos_t, 折射方向 u_trans
    4. Fresnel 决策：
       - 计算 R = fresnel_unpolarized(n1, n2, cos_i, cos_t)
       - 抽取随机数 ξ = rng.random()
       - 若 ξ < R → 反射（镜面对称），返回 "reflected"
       - 否则 → 透射：
         * 若目标是外界：
             - 上方：更新 tallies.R_d += photon.w，photon.alive=False，返回 "exit_top"
             - 下方：更新 tallies.T_d += photon.w，photon.alive=False，返回 "exit_bottom"
         * 若目标是内部层：
             - photon.u = u_trans（单位化）
             - photon.layer_idx = j
             - photon.z 沿新 uz 微小推进 (±1e-12)，避免浮点数误判
             - 返回 "transmitted"

    数值注意事项
    ------------
    - 反射时只需翻转 uz 分量 (ux, uy 不变)。
    - 每次更新方向后建议归一化向量。
    - 出射时不改变 photon.w（不分裂权重）。
    - tallies 为空时需判空，避免报错。
    """    
    
