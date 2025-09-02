import pytest
import numpy as np
from sandbox.minicore_cpu import (
    Photon, 
    step_length, 
    absorb, 
    scatter_isotropic,
)

# ---------- 1) 自由程采样统计 ----------


def test_step_length_stats(tol=0.05):
# """验证 step_length 的统计特性（均值≈1/mu_t）。
# Arrange: 选择多个 mu_t，设置 N 次采样
    mu_t = 2.0
    N = 1000
    rng = np.random.RandomState(0)# U[0,1)

# Act: 采样并计算均值
    samples = [step_length(mu_t, rng) for _ in range(N)]
    mean_val = np.mean(samples)
# Assert: 与理论值在容差内
    assert np.isclose(mean_val, 1.0/mu_t, rtol=tol)


# """
# TODO: 由你实现
# pass

def test_step_length_invalid_mu_t():
# """mu_t <= 0 时应当报错（健壮性）。"""
# # Arrange: 输入异常的mu_t
    mu_t = -1

# Act: 查看是否抛出异常
    with pytest.raises(ValueError):
        _ = step_length(mu_t)

# Assert: 抛出异常




# ---------- 2) 吸收-only 能量守恒 ----------


def test_absorption_only_energy_conservation(tol=0.05):
    

# """当 mu_s=0 时，所有权重应被吸收（A/N ≈ 1）。"""
# # Arrange: 设置mu_s=0, 即mu_a = mu_t
    mu_a = 1
    mu_t = 1
    test_photon = Photon([0,0,0],[0.5,0.5,0])


# Act: 计算吸收权重: Δw = w * (μa/μt)
    dw = absorb(test_photon, mu_a, mu_t)
# Assert: 
    assert dw == 1

# TODO: 由你实现


@pytest.mark.parametrize(
    "mu_a, mu_t, w0, expected_dw, expected_w",
    [
        (0.0, 1.0, 1.0, 0.0, 1.0),   # Case 1: 没有吸收
        (1.0, 1.0, 1.0, 1.0, 0.0),   # Case 2: 全吸收
        (0.2, 1.0, 1.0, 0.2, 0.8),   # Case 3: 部分吸收
    ]
)
def test_absorb_cases(mu_a, mu_t, w0, expected_dw, expected_w):
    photon = Photon([0,0,0],[0,0,1],w = w0)
    dw = absorb(photon, mu_a, mu_t)
    
    #断言1
    assert np.isclose(dw, expected_dw,atol=1e-6)

    #断言2
    assert np.isclose(photon.w, expected_w, atol=1e-6)




# ---------- 3) 各向同性散射统计 ----------


# def test_isotropic_scatter_unit_norm_and_zero_mean(tol, rng):
# # """方向向量单位长度 & 方向分量期望≈0（各向同性）。"""
# # TODO: 由你实现
#     pass


# ---------- 4) 进阶（可选）：参数化、慢测标记 ----------
# @pytest.mark.parametrize("mu_t", [0.5, 1.0, 3.0])
# @pytest.mark.slow
# def test_step_length_parametrized(mu_t, tol):
# # """参数化示例：多组 mu_t 的统计验证（可标记为 slow）。"""
# # TODO: 由你实现
#     pass

