"""
extras/tx_analysis.py
---------------------
T_X (well-switching time) analysis for the Delta-Kerr-Cat qubit.

Computes T_X = kappa_1 / (Liouvillian gap) across sweeps of P/K and/or Delta/K.
Not wired into the main CLI menu — run this file directly.

Usage
-----
    python extras/tx_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from core import params as p
from core.system import KerrCatSystem


# ─────────────────────────────────────────────────────────────────────────────
# Worker (module-level for pickling on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_point(args):
    """
    Compute T_X * kappa_1 for one (P/K, delta/K) point.
    Returns (P_over_K, delta_over_K, T_X_kappa, mixing_rate).
    """
    P_over_K, delta_over_K, N, n_th = args

    sys_  = KerrCatSystem(P_over_K, delta_over_K, N, n_th)
    L     = sys_.liouvillian

    try:
        evals = L.eigenenergies(sparse=True, sort="low", eigvals=8,
                                tol=0, maxiter=10000, sigma=0)
    except Exception:
        evals = L.eigenenergies(sparse=False)

    kappa_over_K = p.kappa_1 / p.K
    threshold    = max(kappa_over_K * 1e-7, 1e-10)
    real_parts   = np.sort(np.abs(evals.real))
    non_zero     = real_parts[real_parts > threshold]

    if len(non_zero) == 0:
        return P_over_K, delta_over_K, np.nan, np.nan

    mixing_rate = non_zero[0]
    T_X         = kappa_over_K / mixing_rate
    return P_over_K, delta_over_K, T_X, mixing_rate


def _hilbert_size(n_bar, N_max=80):
    return min(max(30, int(5 * n_bar) + 15), N_max)


def _run_parallel(tasks):
    n_workers = max(1, cpu_count() - 1)
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_solve_point, tasks),
                            total=len(tasks), desc="T_X sweep"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sweep functions
# ─────────────────────────────────────────────────────────────────────────────

def sweep_vs_P(
    delta_over_K: float = 6.0,
    n_th:         float = 0.05,
    P_min:        float = 0.5,
    P_max:        float = 12.0,
    n_points:     int   = 30,
    N_max:        int   = 60,
):
    """T_X vs P/K at fixed Delta/K."""
    P_vals = np.linspace(P_min, P_max, n_points)
    tasks  = [(P, delta_over_K, _hilbert_size(P, N_max), n_th) for P in P_vals]
    res    = _run_parallel(tasks)

    P_arr  = np.array([r[0] for r in res])
    T_X    = np.array([r[2] for r in res])
    T_X_ms = T_X / (p.kappa_1 / p.K * p.K) * 1e3

    plt.figure(figsize=(7, 5))
    plt.semilogy(P_arr, T_X_ms, "b.-", ms=6)
    plt.xlabel(r"$P/K$", fontsize=13)
    plt.ylabel(r"$T_X$ (ms)", fontsize=13)
    plt.title(rf"$T_X$ vs $P/K$  ($\Delta/K={delta_over_K}$, $n_{{th}}={n_th}$)",
              fontsize=12)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/TX_vs_P_delta{delta_over_K}.png", dpi=200)
    plt.show()
    return P_arr, T_X


def sweep_vs_delta(
    P_over_K:    float = 2.17,
    n_th:        float = 0.05,
    delta_min:   float = 0.0,
    delta_max:   float = 10.0,
    n_points:    int   = 60,
    N_max:       int   = 80,
):
    """T_X vs Delta/K at fixed P/K."""
    d_vals = np.linspace(delta_min, delta_max, n_points)
    tasks  = [(P_over_K, d, _hilbert_size(abs(d) + 2 * P_over_K, N_max), n_th)
              for d in d_vals]
    res    = _run_parallel(tasks)

    d_arr  = np.array([r[1] for r in res])
    T_X    = np.array([r[2] for r in res])
    T_X_ms = T_X / (p.kappa_1 / p.K * p.K) * 1e3

    plt.figure(figsize=(7, 5))
    plt.semilogy(d_arr, T_X_ms, "ko-", ms=4)
    plt.xlabel(r"$\Delta/K$", fontsize=13)
    plt.ylabel(r"$T_X$ (ms)", fontsize=13)
    plt.title(rf"$T_X$ vs $\Delta/K$  ($P/K={P_over_K}$, $n_{{th}}={n_th}$)",
              fontsize=12)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/TX_vs_delta_P{P_over_K}.png", dpi=200)
    plt.show()
    return d_arr, T_X


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    os.makedirs("output", exist_ok=True)
    sweep_vs_P(delta_over_K=6.0, n_th=0.05)
    sweep_vs_delta(P_over_K=2.17, n_th=0.05)
