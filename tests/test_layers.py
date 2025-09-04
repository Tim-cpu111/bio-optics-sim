import pytest
import numpy as np

# 等待实现: from biooptics.models.layers import Layer, LayerStack
def make_two_layer_stack():
    """简单的两层模型:
    layer0: d=0.5, mu_a=0.1, mu_s=1.0, g=0.9, n=1.37
    layer1: d=∞,   mu_a=0.2, mu_s=2.0, g=0.8, n=1.4
    """
    from biooptics.models.layers import Layer, LayerStack
    layers = [
        Layer(mu_a=0.1, mu_s=1.0, g=0.9, n=1.37, d=0.5),
        Layer(mu_a=0.2, mu_s=2.0, g=0.8, n=1.4, d=np.inf),
    ]
    return LayerStack(layers)

def test_find_layer_index():
    stack = make_two_layer_stack()
    # 在第0层内部
    assert stack.find_layer(0.1) == 0
    assert stack.find_layer(0.49) == 0
    # 在界面之下 (z=0.5) 应该进入 layer1
    assert stack.find_layer(0.5) == 1
    # 深层位置
    assert stack.find_layer(10.0) == 1

def test_boundary_distance_downward():
    stack = make_two_layer_stack()
    # 在layer0内，向下走 (uz>0)
    s_b = stack.get_boundary_distance(z=0.2, uz=1.0, idx=0)
    # 应该走到 z=0.5 边界 → s_b≈0.3
    assert pytest.approx(s_b, rel=1e-6) == 0.3

def test_boundary_distance_upward():
    stack = make_two_layer_stack()
    # 在layer0内，向上走 (uz<0)
    s_b = stack.get_boundary_distance(z=0.2, uz=-1.0, idx=0)
    # 应该走到 z=0 顶面 → s_b≈0.2
    assert pytest.approx(s_b, rel=1e-6) == 0.2