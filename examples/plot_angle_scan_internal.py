# examples/plot_angle_scan_internal.py
import numpy as np
import matplotlib.pyplot as plt

from biooptics.models.layers import Layer, LayerStack
from biooptics.mc.photon_types import PhotonRecord
from biooptics.mc.kernels_cpu import snell_refract, fresnel_unpolarized, handle_boundary

class RNGDet:
    def __init__(self, seed=0): self._rng = np.random.default_rng(seed)
    def random(self): return float(self._rng.random())

def simulate_angle(theta_deg, n1=1.37, n2=1.0, trials=8000, seed=0):
    stack = LayerStack([Layer(mu_a=0.0, mu_s=0.0, g=0.0, n=n1, d=np.inf)])
    rng = RNGDet(seed)
    Rd_sum = 0.0

    theta = np.deg2rad(theta_deg)
    uz0 = -float(np.cos(theta))
    ux0 =  float(np.sin(theta)); uy0 = 0.0

    for _ in range(trials):
        p = np.zeros((), dtype=PhotonRecord)
        p["x"]=0.0; p["y"]=0.0; p["z"]=0.0
        p["ux"]=ux0; p["uy"]=uy0; p["uz"]=uz0
        p["w"]=1.0; p["alive"]=1; p["layer_idx"]=0
        tallies = type("T", (), {"R_d":0.0, "T_d":0.0})()
        _ = handle_boundary(p, stack, rng, tallies)
        Rd_sum += tallies.R_d

    Rd_mc = Rd_sum / trials

    # 理论：内部反射率 → 顶出射=1-R
    u_inc = np.array([ux0,0.0,abs(uz0)], dtype=float)
    tir, _, cos_i, cos_t = snell_refract(u_inc, n1, n2)
    if tir: R_int = 1.0
    else:   R_int = fresnel_unpolarized(n1, n2, cos_i, cos_t)
    Rd_th = 1.0 - R_int
    return Rd_mc, Rd_th

def main():
    thetas = np.arange(0, 75+1, 5)
    mc = []; th = []
    for deg in thetas:
        r_mc, r_th = simulate_angle(deg, trials=6000, seed=deg+7)
        mc.append(r_mc); th.append(r_th)

    plt.figure()
    plt.plot(thetas, mc, marker='o', label='MC top-exit (tissue→air)')
    plt.plot(thetas, th, linestyle='--', label='Theory 1 − R_internal')
    plt.xlabel('Incident angle inside tissue (deg)')
    plt.ylabel('Top exit probability')
    plt.title('Internal Fresnel check (μa=μs=0)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
