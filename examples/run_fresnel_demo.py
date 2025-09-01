from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_with_scattering

if __name__ == "__main__":
    # 典型组织参数
    props = TissueOpticalProps(mu_a=0.1, mu_s=10.0, g=0.9, n=1.37)
    R_d, A = simulate_with_scattering(N=50000, props=props, seed=123)
    print(f"R_d = {R_d:.4f}, A = {A:.4f}, sum = {R_d + A:.4f}")
