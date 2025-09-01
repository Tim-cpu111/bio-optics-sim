from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_absorption_only

if __name__ == "__main__":
    props = TissueOpticalProps(mu_a=1.0, mu_s=0.0, g=0.0, n=1.37)
    R_d, A = simulate_absorption_only(N=5000, props=props)
    print(f"反射率 R_d = {R_d:.3f}, 吸收率 A = {A:.3f}, 能量守恒 = {R_d + A:.3f}")
