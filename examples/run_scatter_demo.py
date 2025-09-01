from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_with_scattering

if __name__ == "__main__":
    for g in (0.0, 0.5, 0.9):
        props = TissueOpticalProps(mu_a=0.2, mu_s=5.0, g=g, n=1.37)
        R_d, A = simulate_with_scattering(N=20000, props=props, seed=42)
        print(f"g={g:.1f} -> R_d={R_d:.3f}, A={A:.3f}, R_d+A={R_d+A:.3f}")

