from dataclasses import dataclass

@dataclass
class TissueOpticalProps:
    mu_a: float   # 吸收系数 [1/mm]
    mu_s: float   # 散射系数 [1/mm]
    g: float      # 各向异性因子
    n: float      # 折射率

    @property
    def mu_t(self) -> float:
        return self.mu_a + self.mu_s

