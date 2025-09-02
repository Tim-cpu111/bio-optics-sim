# src/biooptics/models/layers.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Layer:
    """描述单层组织的光学参数"""
    mu_a: float   # 吸收系数
    mu_s: float   # 散射系数
    g: float      # 各向异性因子
    n: float      # 折射率
    d: float      # 厚度 (np.inf 表示半无限)

class LayerStack:
    """多层组织堆栈，提供层索引和边界距离计算"""

    def __init__(self, layers):
        self.layers = layers
        # 累计厚度边界，比如 [0.5, ∞]
        self.boundaries = np.cumsum([layer.d for layer in layers])

    def find_layer(self, z: float) -> int:
        """给定光子位置 z，返回所在层索引"""
        for idx, boundary in enumerate(self.boundaries):
            if z < boundary:
                return idx
        # 如果超出所有边界，默认最后一层
        return len(self.layers) - 1
       

    def get_boundary_distance(self, z: float, uz: float, idx: int) -> float:
        """计算当前位置到最近边界的距离"""
        raise NotImplementedError
