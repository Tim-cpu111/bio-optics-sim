import numpy as np

def rng(seed: int = None):
    """统一的随机数接口"""
    return np.random.default_rng(seed)
