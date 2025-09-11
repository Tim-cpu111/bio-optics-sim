# examples/plot_convergence_small.py
import numpy as np
import matplotlib.pyplot as plt

from biooptics.models.layers import Layer, LayerStack
from biooptics.simulation.run_small import run_small, SmallSimConfig

def main():
    stack = LayerStack([
        Layer(mu_a=0.1, mu_s=10.0, g=0.9, n=1.37, d=1.0),
        Layer(mu_a=0.0, mu_s=0.0,  g=0.0, n=1.50, d=np.inf),
    ])
    Ns = np.unique(np.round(np.logspace(2, 4, 10)).astype(int))  # 1e2..1e4
    totals = []
    for n in Ns:
        cfg = SmallSimConfig(n_photons=int(n), rng_seed=42)
        Rd, Td, A = run_small(stack, cfg)
        totals.append(Rd + Td + A.sum())

    plt.figure()
    plt.plot(Ns, totals, marker='o')
    plt.axhline(1.0, linestyle='--')
    plt.xscale('log')
    plt.xlabel('Photon count'); plt.ylabel('R + T + ΣA')
    plt.title('Energy conservation convergence')
    plt.grid(True, which='both'); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
