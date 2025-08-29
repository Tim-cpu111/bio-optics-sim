# examples/visualize_s1.py
import math
import numpy as np
import matplotlib.pyplot as plt

from biooptics.models.materials import TissueOpticalProps
from biooptics.mc.kernels_cpu import simulate_with_scattering
from biooptics.mc.tallies import Tallies


def collect_single_run(props: TissueOpticalProps, N=8000, seed=42,
                       track_trajectories=True, max_tracks=60):
    """跑一次 MC，采集步长、uz、轨迹与 R_d/A。"""
    t = Tallies(track_trajectories=track_trajectories, max_tracks=max_tracks)
    R_d, A = simulate_with_scattering(
        N=N, props=props, seed=seed, tallies=t,
        rr_threshold=1e-3, rr_p=0.1
    )
    return t.sampled_steps, t.sampled_uz, t.tracks, (R_d, A)


def energy_convergence(props: TissueOpticalProps, Ns=(200, 500, 1000, 5000, 10000, 20000), seed=123):
    """计算 |R_d + A - 1| 随 N 的收敛曲线（当前阶段 R_d≈0）。"""
    errs = []
    for N in Ns:
        t = Tallies(track_trajectories=False)
        R_d, A = simulate_with_scattering(N=N, props=props, seed=seed, tallies=t)
        errs.append(abs(R_d + A - 1.0))
    return list(Ns), errs


def plot_all_in_one(steps, mu_t, uz, g, tracks, Ns, errs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 1) Step length distribution
    ax = axes[0, 0]
    s = np.asarray(steps, dtype=float)
    if s.size == 0:
        ax.text(0.5, 0.5, "no steps collected", ha="center", va="center")
    else:
        ax.hist(s, bins=60, density=True, alpha=0.6, label="sampled")
        x_max = np.percentile(s, 99.5)
        xs = np.linspace(0.0, max(1e-6, x_max), 300)
        ax.plot(xs, mu_t * np.exp(-mu_t * xs), lw=2, label=r"theory Exp($\mu_t$)")
        ax.set_xlabel("step s (mm)"); ax.set_ylabel("pdf")
        ax.set_title("Step length distribution"); ax.legend()

    # 2) uz distribution
    ax = axes[0, 1]
    uz = np.asarray(uz, dtype=float)
    if uz.size == 0:
        ax.text(0.5, 0.5, "no uz collected", ha="center", va="center")
    else:
        ax.hist(uz, bins=60, density=True, alpha=0.7)
        ax.set_xlabel("uz"); ax.set_ylabel("pdf")
        ax.set_title(f"uz distribution (g={g})")

    # 3) Photon tracks (x–z)
    ax = axes[1, 0]
    any_track = False
    for t in tracks:
        if len(t) > 1:
            x, z = zip(*t)
            ax.plot(x, z, lw=0.7)
            any_track = True
    if not any_track:
        ax.text(0.5, 0.5, "no tracks collected", ha="center", va="center")
    ax.invert_yaxis()  # z 向下
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title("Sample photon tracks (x–z)")

    # 4) Energy conservation convergence
    ax = axes[1, 1]
    ax.plot(Ns, errs, marker="o")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Photons N"); ax.set_ylabel("|R_d + A - 1|")
    ax.set_title("Energy conservation convergence")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 参数：可按需调整
    props_g09 = TissueOpticalProps(mu_a=0.2, mu_s=5.0, g=0.9, n=1.37)
    props_g0  = TissueOpticalProps(mu_a=0.2, mu_s=5.0, g=0.0, n=1.37)

    # 单次运行：采集步长、uz、轨迹（用 g=0.9）
    steps, uz, tracks, (Rd, A) = collect_single_run(props_g09, N=8000, seed=42,
                                                    track_trajectories=True, max_tracks=60)
    print(f"single run (g=0.9): R_d={Rd:.6f}, A={A:.6f}, R_d+A={Rd+A:.6f}")

    # 收敛曲线：用 g=0（各向同性）做演示
    Ns, errs = energy_convergence(props_g0, Ns=(200, 500, 1000, 5000, 10000, 20000), seed=123)

    # 合成大图
    fig = plot_all_in_one(steps, props_g09.mu_t, uz, props_g09.g, tracks, Ns, errs)
    plt.show()
