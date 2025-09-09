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
       
    def len_layer(self):
        return len(self.layers)

    def get_boundary_distance(self, z: float, uz: float, idx: int) -> float:
        """
        计算当前位置到最近层界的距离（沿当前方向的步长 s_b）。
        约定：uz>0 向下撞下边界；uz<0 向上撞上边界；uz=0 则永不撞界（返回 inf）。
        """


        esp = 1e-12
        if abs(uz) < esp:
            return np.inf
        
        #当前层的上下边界
        z_top = 0.0 if idx == 0 else self.boundaries[idx - 1]
        z_bot = self.boundaries[idx]

        if uz > 0.0:
            #向下：目标是下边界
            if np.isinf(z_bot):
                return np.inf
        
            s = (z_bot - z)/uz

        else:
            #向上：目标是上边界
            s = (z - z_top)/(-uz)

        # 数值稳健：不允许返回负步长
        if s < 0.0 and s > -1e-9:
            s =  0.0
        
        return s if s >= 0.0 else np.inf 
