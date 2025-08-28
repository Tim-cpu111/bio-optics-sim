import numpy as np

# 定义单个光子的数据结构
PhotonRecord = np.dtype([
    ("x","f4"),("y","f4"),("z","f4"),    # 位置
    ("ux","f4"),("uy","f4"),("uz","f4"), # 方向余弦
    ("w","f4"),                          # 权重
    ("alive","i1")                       # 是否存活 (1=活, 0=死)
])
