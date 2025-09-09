import pytest
import numpy as np
from biooptics.mc.kernels_cpu import snell_refract, fresnel_unpolarized, handle_boundary
from numpy import  deg2rad
"""
tests/test_fresnel.py

测试 Fresnel 与 Snell 两个底层函数:
- 几何正确性: 折射、全反射是否符合 Snell 定律。
- 数值正确性: Fresnel 反射率在正入射极限与范围 [0,1] 内。
"""

def test_snell_refraction_geometry_non_tir():
    # 空气->组织：n1 < n2，选一个中等入射角，必非 TIR
    n1, n2 = 1.0, 1.37
    theta_i = deg2rad(30.0)
    uz = float(np.cos(theta_i))
    ut = float(np.sin(theta_i))
    u_incident = np.array([ut, 0.0, uz], dtype=float)  # 朝下撞下界

    tir, u_trans, cos_i, cos_t = snell_refract(u_incident, n1, n2)

    # 1) 不应发生全反射
    assert tir is False
    # 2) 返回有效折射向量和余弦
    assert u_trans is not None and cos_t is not None
    # 3) 折射向量单位化
    assert pytest.approx(np.linalg.norm(u_trans), rel=1e-12) == 1.0
    # 4) Snell 验证：n1*sin(theta_i) ≈ n2*sin(theta_t)
    #   sin(theta_i) = sqrt(1 - cos_i^2), sin(theta_t) = sqrt(1 - cos_t^2)
    lhs = n1 * np.sqrt(max(0.0, 1.0 - float(cos_i)**2))
    rhs = n2 * np.sqrt(max(0.0, 1.0 - float(cos_t)**2))
    assert pytest.approx(lhs, rel=1e-12, abs=1e-12) == rhs
    # 5) 折射后的 z 分量应朝传播方向（这里仍向下）
    assert u_trans[2] > 0.0

def test_snell_tir_condition():
    """验证超过临界角时，snell_refract 返回 tir=True 且无折射向量。"""
    n1, n2 = 1.5, 1.0
    theta_c = np.arcsin(n2/n1) #~0.7297 rad
    theta_i = theta_c + deg2rad(5.0)

    # 选择“朝下”入射（撞下边界）：uz = +cos(theta_i)
    uz = float(np.cos(theta_i))
    ut = float(np.sin(theta_i))
    # 把切向量放在 x 方向上即可（|u|=1）
    u_incident = np.array([ut, 0.0, uz], dtype=float)

    tir, u_trans, cos_i, cos_t = snell_refract(u_incident, n1, n2)

    assert tir is True
    assert u_trans is None
    assert cos_t is None
    # cos_i 应为 |u_z|
    assert pytest.approx(cos_i, rel=1e-12) == abs(uz)





def test_fresnel_normal_incidence_limit():
    
    """验证法向入射时 Fresnel 反射率退化为 (n1-n2)^2/(n1+n2)^2。"""

    n1, n2 = 1.0, 1.37
    cos_i = 1.0
    cos_t = 1.0
    R = fresnel_unpolarized(n1, n2, cos_i, cos_t)
    R0 = ((n1 - n2) / (n1 + n2))**2
    assert R == pytest.approx(R0, rel=1e-12)


def test_fresnel_value_range():
    n1, n2 = 1.33, 1.4
    for deg in [0, 15, 30, 45, 60]:
        ci = float(np.cos(np.deg2rad(deg)))
        # 用 Snell 算 cos_t
        sin_t = (n1/n2) * np.sqrt(max(0.0, 1.0 - ci**2))
        if sin_t >= 1.0:  # 超过临界角，跳过
            continue
        ct = np.sqrt(1.0 - sin_t**2)
        R = fresnel_unpolarized(n1, n2, ci, ct)
        assert 0.0 <= R <= 1.0
