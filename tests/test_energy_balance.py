# tests/test_energy_balance.py
import numpy as np
import pytest

from biooptics.models.layers import Layer, LayerStack
from biooptics.simulation.run_small import run_small, SmallSimConfig  # ← 先写测试：当前还不存在

def _two_layer_stack():
    # layer0：组织；layer1：玻璃/高 n 层（示意），底部半无限
    layers = [
        Layer(mu_a=0.1, mu_s=10.0, g=0.9, n=1.37, d=1.0),
        Layer(mu_a=0.0, mu_s=0.0,  g=0.0, n=1.50, d=np.inf),
    ]
    return LayerStack(layers)


def test_energy_conservation_small_run():
    stack = _two_layer_stack()
    cfg = SmallSimConfig(n_photons=2000, rng_seed=2025)
    Rd, Td, A_layers = run_small(stack, cfg)

    total = Rd + Td + float(np.sum(A_layers))
    assert np.isclose(total, 1.0, atol=2e-2), f"Energy not conserved: {total:.4f}"

    # 基本合理性
    assert Rd >= 0 and Td >= 0
    assert A_layers.shape == (len(stack.layers),)
    assert np.all(A_layers >= 0)

def test_reproducible_with_seed():
    
    stack = _two_layer_stack()
    cfg1 = SmallSimConfig(n_photons=1000, rng_seed=7)
    cfg2 = SmallSimConfig(n_photons=1000, rng_seed=7)

    Rd1, Td1, A1 = run_small(stack, cfg1)
    Rd2, Td2, A2 = run_small(stack, cfg2)

    assert Rd1 == pytest.approx(Rd2, rel=0, abs=1e-12)
    assert Td1 == pytest.approx(Td2, rel=0, abs=1e-12)
    assert np.allclose(A1, A2, atol=1e-12)