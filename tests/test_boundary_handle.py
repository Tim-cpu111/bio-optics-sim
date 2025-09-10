# tests/test_boundary_handle.py
import pytest
import numpy as np

# 目标：让这些测试现在就“红灯”但可运行，等实现后再改断言
from biooptics.mc.kernels_cpu import handle_boundary
from biooptics.models.layers import Layer, LayerStack
from biooptics.mc.photon_types import PhotonRecord


from types import SimpleNamespace


def _make_test_stack():
    """
    两层模型：
    - layer0：有限厚（d=0.5），n=1.37
    - layer1：半无限（d=inf），n=1.40
    顶面在 z=0，z 轴向下为正。
    """
    layers = [
        Layer(mu_a=0.1, mu_s=1.0, g=0.9, n=1.37, d=0.5),
        Layer(mu_a=0.2, mu_s=2.0, g=0.8, n=1.4, d=np.inf),
    ]
    return LayerStack(layers)

def _make_test_photon_TIR_at_top(theta_deg=60.0, w=1.0):

    """
    构造一个在顶界面发生全反射（TIR）的光子：
    - 位置 z=0（顶面），处于 layer0（idx=0）
    - 指向“上方”（uz<0），入射角 theta > theta_c，使得从 n1=1.37 入射到外界 n2=1.0 发生 TIR
    例如 theta=60°，临界角大约是 arcsin(1.0/1.37)≈47.3°
    """
    theta = np.deg2rad(theta_deg)
    uz = -float(np.cos(theta))   # 向上：uz<0
    ux = float(np.sin(theta))    # 切向放在 x 轴即可
    uy = 0.0

    p = np.zeros((), dtype=PhotonRecord)
    p["x"], p["y"], p["z"] = 0.0, 0.0, 0.0
    p["ux"], p["uy"], p["uz"] = ux, uy, uz
    p["w"] = w
    p["layer_idx"] = 0
    p["alive"] = 1
    return p

def _make_test_photon_no_TIR_at_top(theta_deg=30, w=1.0):

    """
    构造一个在顶界面发生全反射（TIR）的光子：
    - 位置 z=0（顶面），处于 layer0（idx=0）
    - 指向“上方”（uz<0），入射角 theta > theta_c，使得从 n1=1.37 入射到外界 n2=1.0 发生 TIR
    例如 theta=60°，临界角大约是 arcsin(1.0/1.37)≈47.3°
    """
    theta = np.deg2rad(theta_deg)
    uz = float(np.cos(theta))   # 向上：uz<0
    ux = float(np.sin(theta))    # 切向放在 x 轴即可
    uy = 0.0

    p = np.zeros((), dtype=PhotonRecord)
    p["x"], p["y"], p["z"] = 0.0, 0.0, 0.5
    p["ux"], p["uy"], p["uz"] = ux, uy, uz
    p["w"] = w
    p["layer_idx"] = 0
    p["alive"] = 1
    return p


def test_handle_boundary_reflect_mirror_direction():
    """测试全反射或高 Fresnel 反射率场景：
    期望光子在界面镜面反射，uz 分量取反，
    layer_idx 保持不变，方向向量仍为单位向量。"""


    photon = _make_test_photon_TIR_at_top(theta_deg=60.0, w=1.0)
    stack = _make_test_stack()
    rng = np.random.default_rng(123)

    # 反射前记录 —— 立刻转成 Python float，避免后续负号/比较混乱
    ux_before = float(photon["ux"])
    uy_before = float(photon["uy"])
    uz_before = float(photon["uz"])

    tallies = SimpleNamespace(R_d=0.0, T_d=0.0)
    
    #结果
    outcome = handle_boundary(photon, stack, rng, tallies)
    assert outcome == "reflected"

    # 比较时，把当前值也转成 float
    assert pytest.approx(float(photon["ux"]), rel=1e-12) == ux_before
    assert pytest.approx(float(photon["uy"]), rel=1e-12) == uy_before
    assert pytest.approx(float(photon["uz"]), rel=1e-6) == -uz_before



    # 3) 方向仍为单位向量（防止数值漂移）
    norm = np.sqrt(photon["ux"]**2 + photon["uy"]**2 + photon["uz"]**2)
    assert pytest.approx(norm, rel=1e-6) == 1

    # 4) 层不变
    
    
    assert pytest.approx(float(photon["layer_idx"]), rel=1e-12) == 0

    # 5) 光子仍存活（反射后继续传播）
    assert photon["alive"] == 1



    # ✅ 等实现后，把上面改为：
    # outcome = handle_boundary(photon, stack, rng)
    # assert outcome == "reflected"
    # 还要断言：反射方向为镜面反射（uz 取反，切向分量不变），layer_idx 不变

def test_handle_boundary_transmit_and_layer_change():
    """测试非 TIR 且 Fresnel 反射率较小的场景：
    期望光子透射进入相邻层，layer_idx 改变，
    新方向满足 Snell 定律，且 uz 符号正确。"""
    photon = _make_test_photon_no_TIR_at_top(theta_deg=10, w=1.0)
    stack = _make_test_stack()
    rng = np.random.default_rng(123)
    # 反射前记录 —— 立刻转成 Python float，避免后续负号/比较混乱
    ux_before = float(photon["ux"])
    uy_before = float(photon["uy"])
    uz_before = float(photon["uz"])
    tallies = SimpleNamespace(R_d=0.0, T_d=0.0)
    
    #结果
    outcome = handle_boundary(photon, stack, rng, tallies)
    # 1) 结果为透射
    assert outcome == "transmitted"
        # 比较时，把当前值也转成 float
    # assert pytest.approx(float(photon["ux"]), rel=1e-12) == ux_before
    # assert pytest.approx(float(photon["uy"]), rel=1e-12) == uy_before
    # assert pytest.approx(float(photon["uz"]), rel=1e-6) == -uz_before


    # outcome = handle_boundary(photon, stack, rng)

    # # 3) 方向仍为单位向量（防止数值漂移）
    # norm = np.sqrt(photon["ux"]**2 + photon["uy"]**2 + photon["uz"]**2)
    # assert pytest.approx(norm, rel=1e-6) == 1.0

    # 2) 进入相邻层    
    assert pytest.approx(float(photon["layer_idx"]), rel=1e-12) == 1

    # 3） 方向单位化
    norm = float(np.sqrt(photon["ux"]**2 + photon["uy"]**2 + photon["uz"]**2))
    assert norm == pytest.approx(1.0, rel=1e-12)

    n1 = float(stack.layers[0].n)
    n2 = float(stack.layers[1].n)

    ci = abs(uz_before); ct = abs(photon["uz"])
    lhs = n1*np.sqrt(max(0.0, 1.0-ci**2)); rhs = n2*np.sqrt(max(0.0, 1.0-ct**2))
    assert lhs == pytest.approx(rhs, rel=1e-5)

    # 5) 光子仍存活（反射后继续传播）
    assert photon["alive"] == 1


    # ✅ 等实现后：
    # outcome = handle_boundary(photon, stack, rng)
    # assert outcome == "transmitted"
    # 断言：方向满足 Snell（cos_t 合理、单位向量），layer_idx 正确进入相邻层




class RNGHigh:
    """固定返回大数，保证 xi > R，从而选择透射分支。"""
    def random(self):
        return 0.99


def _make_stack_top_exit():
    """顶面出射：上方为空气（外界），用两层，顶层有限厚即可。"""
    layers = [
        Layer(mu_a=0.1, mu_s=1.0, g=0.9, n=1.37, d=0.5),     # layer0
        Layer(mu_a=0.2, mu_s=2.0, g=0.8, n=1.40, d=np.inf), # layer1 半无限
    ]
    return LayerStack(layers)


def _make_stack_bottom_exit():
    """底面出射：最后一层设为有限厚，这样其下方就是外界。"""
    layers = [
        Layer(mu_a=0.1, mu_s=1.0, g=0.9, n=1.37, d=0.5),  # layer0
        Layer(mu_a=0.2, mu_s=2.0, g=0.8, n=1.40, d=0.3),  # layer1 也有限厚
    ]
    return LayerStack(layers)


def _make_photon_at_top(theta_deg=10.0, w=1.0):
    """放在顶面 z=0，向上（uz<0），小角度，非 TIR。"""
    theta = np.deg2rad(theta_deg)
    uz = -float(np.cos(theta))   # 向上
    ux =  float(np.sin(theta))
    uy =  0.0
    p = np.zeros((), dtype=PhotonRecord)
    p["x"]=0.0; p["y"]=0.0; p["z"]=0.0
    p["ux"]=ux; p["uy"]=uy; p["uz"]=uz
    p["w"]=w;  p["layer_idx"]=0; p["alive"]=1
    return p


def _make_photon_at_bottom(stack, theta_deg=10.0, w=1.0):
    """放在最后一层的底面，向下（uz>0），小角度，非 TIR。"""
    theta = np.deg2rad(theta_deg)
    uz =  float(np.cos(theta))   # 向下
    ux =  float(np.sin(theta))
    uy =  0.0
    # 计算最底层底面的 z：= 所有层厚度之和
    z_bottom = float(np.sum([ly.d for ly in stack.layers]))
    last_idx = len(stack.layers) - 1

    p = np.zeros((), dtype=PhotonRecord)
    p["x"]=0.0; p["y"]=0.0; p["z"]=z_bottom
    p["ux"]=ux; p["uy"]=uy; p["uz"]=uz
    p["w"]=w;  p["layer_idx"]=last_idx; p["alive"]=1
    return p


def test_exit_tallies_top_bottom():
    """验证顶/底出射时的记账与终止：
    - 顶面：outcome='exit_top' 且 R_d 增量≈w，photon.alive=0
    - 底面：outcome='exit_bottom' 且 T_d 增量≈w，photon.alive=0
    使用固定高随机数 RNG 强制走透射分支，避免 Fresnel 概率干扰。
    """
    # 顶面出射
    stack_top = _make_stack_top_exit()
    photon_top = _make_photon_at_top(theta_deg=10.0, w=1.0)
    tallies = SimpleNamespace(R_d=0.0, T_d=0.0)
    rng_high = RNGHigh()

    outcome_top = handle_boundary(photon_top, stack_top, rng_high, tallies=tallies)
    assert outcome_top == "exit_top"
    assert tallies.R_d == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert tallies.T_d == pytest.approx(0.0, rel=1e-12, abs=1e-12)
    assert photon_top["alive"] == 0

    # 底面出射
    stack_bot = _make_stack_bottom_exit()
    photon_bot = _make_photon_at_bottom(stack_bot, theta_deg=10.0, w=1.0)
    # 重置 tally
    tallies.R_d = 0.0
    tallies.T_d = 0.0

    outcome_bot = handle_boundary(photon_bot, stack_bot, rng_high, tallies=tallies)
    assert outcome_bot == "exit_bottom"
    assert tallies.T_d == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert tallies.R_d == pytest.approx(0.0, rel=1e-12, abs=1e-12)
    assert photon_bot["alive"] == 0
