import numpy as np
from ..models.materials import TissueOpticalProps
from .photon_types import PhotonRecord

def simulate_absorption_only(N: int, props: TissueOpticalProps, seed: int = 42):
    """
    半无限均匀介质 - 仅考虑吸收 (不散射, 不边界).
    验证目标: 所有光子最终都被吸收 => A ≈ 1.
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
        w = photons[i]["w"]
        while photons[i]["alive"]:
            # 1) 随机步长
            s = -np.log(rng.random()) / mu_t
            photons[i]["z"] += s * photons[i]["uz"]

            # 2) 吸收
            absorb = w * (mu_a / mu_t)
            A_total += absorb
            w -= absorb

            # 3) 终止条件 (光子权重趋近于0)
            if w < 1e-6:
                photons[i]["alive"] = 0
        photons[i]["w"] = w

    # 所有光子最终都被吸收 => A_total / N ≈ 1
    return 0.0, A_total / N   # 返回 R_d, A
