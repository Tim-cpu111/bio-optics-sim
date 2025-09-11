import numpy as np
import matplotlib.pyplot as plt

from biooptics.models.layers import Layer, LayerStack
from biooptics.simulation.run_small import run_small, SmallSimConfig
from biooptics.mc.tallies import VizTallies

def main():
    stack = LayerStack([
        Layer(mu_a=0.1, mu_s=10.0, g=0.9, n=1.37, d=1),
        Layer(mu_a=0.0, mu_s=0.0,  g=0.0, n=1.50, d=np.inf),
    ])
    viz = VizTallies(track_trajectories=True, max_points=50000, step_stride=1)
    cfg = SmallSimConfig(n_photons=5000)
    Rd, Td, A = run_small(stack, cfg, viz=viz, viz_first_photons=200)
    # 1) 步长分布
    steps = np.array(viz.sampled_steps, dtype=float)
    if len(steps) > 0:
        plt.figure()
        n, bins, _ = plt.hist(steps, bins=60, density=True, alpha=0.6, label="sampled")
        # 简单对比指数分布形状（取近似平均 1/<s> ~ mu_t 的层内值，这里仅做形状展示）
        plt.title("Step length distribution"); plt.xlabel("step s (mm)"); plt.ylabel("pdf")
        plt.legend(); plt.tight_layout()

    # 2) uz 分布（HG）
    uzs = np.array(viz.sampled_uz, dtype=float)
    if len(uzs) > 0:
        plt.figure()
        plt.hist(uzs, bins=60, density=True, alpha=0.8)
        plt.title("uz distribution"); plt.xlabel("uz"); plt.ylabel("pdf"); plt.tight_layout()

    # 3) 轨迹（x-z）
    if viz.tracks and len(viz.tracks[0]) > 0:
        xs, zs = zip(*viz.tracks[0])
        plt.figure()
        plt.plot(xs, zs, linewidth=0.7, alpha=0.8)
        plt.gca().invert_yaxis()  # z向下为正时可视化更直观
        plt.title("Sample photon track (x–z)")
        plt.xlabel("x (mm)"); plt.ylabel("z (mm)"); plt.tight_layout()

    # 4) 打印守恒
    print(f"R_d={Rd:.4f}, T_d={Td:.4f}, ΣA={A.sum():.4f}, total={Rd+Td+A.sum():.4f}")

    plt.show()

if __name__ == "__main__":
    main()
