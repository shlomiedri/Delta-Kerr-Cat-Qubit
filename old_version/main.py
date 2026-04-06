from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import physical as phys
import params as g


# ──────────────────────────────────────────────────────────────────────────────
# Hilbert space sizing
# ──────────────────────────────────────────────────────────────────────────────

def _hilbert_size(n_bar, N_max):
    return min(max(g.N, int(5 * n_bar) + 15), N_max)


def _hilbert_size_delta(delta, P_val, N_max):
    """Larger N needed when delta shifts photon distribution."""
    n_bar_approx = (abs(delta) + 2 * P_val) / g.K
    return min(max(30, int(n_bar_approx) + 40), N_max)


# ──────────────────────────────────────────────────────────────────────────────
# Core solvers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_mixing_rate(eigvals, kappa1_over_K):
    """Smallest |Re(λ)| above flat noise guard = well-switching rate."""
    threshold  = max(kappa1_over_K * 1e-7, 1e-10)
    real_parts = np.sort(np.abs(eigvals.real))
    non_zero   = real_parts[real_parts > threshold]
    return non_zero[0] if len(non_zero) else np.nan


def _liouvillian_eigvals(L):
    """Shift-invert sparse solve around 0 — fast; fallback to dense if needed."""
    try:
        return L.eigenenergies(
            sparse=True, sort='low', eigvals=8,
            tol=0, maxiter=10000, sigma=0)
    except Exception:
        return L.eigenenergies(sparse=False)


def _solve_point(args):
    """
    Compute T_X·κ₁ and gap for one (P, delta) point.
    args = (P, delta, kappa1_over_K, n_th, N)
    Returns (P/K, delta/K, T_X·κ₁, gap).
    """
    P, delta, kappa1_over_K, n_th, N = args

    print(f"  [worker] P/K={P/g.K:.2f}  Δ/K={delta/g.K:.2f}  n_th={n_th}  N={N}", flush=True)
    ops  = phys.make_operators(N=N)
    H    = phys.make_hamiltonian(ops, P=P, K=g.K, delta=delta)
    c_op = phys.make_collapse_ops(ops, kappa1=kappa1_over_K * g.K, nth=n_th)
    L    = qt.liouvillian(H, c_ops=c_op)

    ev          = _liouvillian_eigvals(L)
    mixing_rate = _extract_mixing_rate(ev, kappa1_over_K)
    T_X         = kappa1_over_K / mixing_rate if not np.isnan(mixing_rate) else np.nan
    return (P / g.K, delta / g.K, T_X, mixing_rate)


def _run_parallel(arg_list, n_workers):
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(_solve_point, arg_list),
            total=len(arg_list),
            desc="T_X points",
            unit="pt",
        ))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_fig(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"Saved: {fname}")


def _print_results(results):
    for r in results:
        print(f"  P/K={r[0]:.2f}  Δ/K={r[1]:.2f}  "
              f"gap={r[3]:.3e}  T_X·κ₁={r[2]:.4e}")


def _ms(T_X_vals, kappa1_over_K):
    return T_X_vals / (kappa1_over_K * g.K) * 1e3

def _compute_T_X_vs_P(delta_val, kappa1_over_K, n_th, P_min_over_K, P_max_over_K,
                      n_points, N_max, n_workers):
    P_vals   = np.linspace(P_min_over_K * g.K, P_max_over_K * g.K, n_points)
    arg_list = [(P, delta_val, kappa1_over_K, n_th, _hilbert_size(P / g.K, N_max))
                for P in P_vals]

    print(f"\n[compute T_X vs P] {n_points} pts | κ₁/K={kappa1_over_K:.2e} | "
          f"n_th={n_th} | Δ={delta_val/g.K:.1f}K | workers={n_workers}")

    results  = _run_parallel(arg_list, n_workers)
    _print_results(results)

    P_norm   = np.array([r[0] for r in results])
    T_X_vals = np.array([r[2] for r in results])
    gap_vals = np.array([r[3] for r in results])
    return P_norm, T_X_vals, gap_vals, _ms(T_X_vals, kappa1_over_K)


def _compute_T_X_vs_delta(P_val, kappa1_over_K, n_th, delta_min_K, delta_max_K,
                           n_points, N_max, n_workers):
    """Return (d_norm, T_X_vals, T_X_ms) for a sweep over Δ/K."""
    delta_vals = np.linspace(delta_min_K * g.K, delta_max_K * g.K, n_points)
    arg_list   = [(P_val, d, kappa1_over_K, n_th,
                   _hilbert_size_delta(d, P_val, N_max))
                  for d in delta_vals]

    print(f"\n[compute T_X vs Δ] {n_points} pts | κ₁/K={kappa1_over_K:.2e} | "
          f"n_th={n_th} | ε₂/K={P_val/g.K:.2f} | workers={n_workers}")

    results  = _run_parallel(arg_list, n_workers)
    _print_results(results)

    d_norm   = np.array([r[1] for r in results])
    T_X_vals = np.array([r[2] for r in results])
    return d_norm, T_X_vals, _ms(T_X_vals, kappa1_over_K)



def plot_T_X_vs_P(
    delta_val     = 0.0,
    kappa1_over_K = None,
    n_th          = g.n_th,
    P_min_over_K  = 0.5,
    P_max_over_K  = 10.0,
    n_points      = 30,
    N_max         = 60,
    n_workers     = None,
):
    """Single-curve semilogy of T_X (ms) vs P/K at fixed Δ and n_th."""
    if kappa1_over_K is None:
        kappa1_over_K = g.kappa_1 / g.K
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    P_norm, T_X_vals, gap_vals, T_X_ms = _compute_T_X_vs_P(
        delta_val, kappa1_over_K, n_th,
        P_min_over_K, P_max_over_K, n_points, N_max, n_workers)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        rf'$T_X$ vs $P/K$ — $\kappa_1/K={kappa1_over_K:.1e}$, '
        rf'$\Delta={delta_val/g.K:.0f}K$, $n_{{th}}={n_th}$', fontsize=13)

    ax.semilogy(P_norm, T_X_ms, 'b.-', markersize=6, lw=1.5)
    ax.set_xlabel(r'$P/ K$')
    ax.set_ylabel(r'$T_X$ (ms)')
    ax.grid(True, which='both', ls=':', alpha=0.5)

    _save_fig(f'T_X_vs_P_kappa{kappa1_over_K:.0e}_delta{delta_val/g.K:.0f}K.png')
    return P_norm, T_X_vals, gap_vals


def plot_T_X_vs_delta(
    P_val         = 2.17 * g.K,
    kappa1_over_K = None,
    n_th          = g.n_th,
    delta_min_K   = 0.0,
    delta_max_K   = 10.0,
    n_points      = 60,
    N_max         = 80,
    n_workers     = None,
):
    """Single-curve semilogy of T_X (ms) vs Δ/K at fixed P and n_th."""
    if kappa1_over_K is None:
        kappa1_over_K = g.kappa_1 / g.K
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    d_norm, T_X_vals, T_X_ms = _compute_T_X_vs_delta(
        P_val, kappa1_over_K, n_th,
        delta_min_K, delta_max_K, n_points, N_max, n_workers)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        rf'$T_X$ vs $\Delta/K$ — $\epsilon_2/K={P_val/g.K:.2f}$, '
        rf'$\kappa_1/K={kappa1_over_K:.1e}$, $n_{{th}}={n_th}$', fontsize=13)

    for ax in axes:
        for m in range(0, int(delta_max_K) + 2, 2):
            ax.axvline(m, color='gray', ls='--', alpha=0.4)
        ax.set_xlabel(r'$\Delta/K$')
        ax.set_xlim(-0.3, delta_max_K + 0.3)
        ax.grid(True, which='both', ls=':', alpha=0.5)

    axes[0].semilogy(d_norm, T_X_ms, 'ko-', markersize=4, lw=1.2)
    axes[0].set(ylabel=r'$T_X$ (ms)', title=r'$T_X$ (ms) — log scale')

    axes[1].semilogy(d_norm, T_X_vals, 'b.-', markersize=4)
    axes[1].set(ylabel=r'$T_X \cdot \kappa_1$',
                title=r'$T_X \cdot \kappa_1$ — log scale')

    _save_fig(f'T_X_vs_Delta_P{P_val/g.K:.2f}_kappa{kappa1_over_K:.0e}.png')
    return d_norm, T_X_vals



def plot_figure_P_versus_Delta(
    delta_list    = (0.0, 2.0, 4.0, 6.0),   # Δ/K values — one curve each
    kappa1_over_K = 1/50,
    n_th          = 0.05,
    P_min_over_K  = 0.5,
    P_max_over_K  = 12.0,
    n_points      = 50,
    N_max         = 80,
    n_workers     = None,
):
    """
    T_X (ms) vs P/K, one semilogy curve per Δ value.
    X-axis: P/K.  Family parameter: Δ/K.
    Calls _compute_T_X_vs_P (= plot_T_X_vs_P computation) for each Δ.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(delta_list)))

    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        rf'$T_X$ vs $\epsilon_2/K$ — $\kappa_1/K={kappa1_over_K:.2g}$, '
        rf'$n_{{th}}={n_th}$', fontsize=12)
    ax.set_xlabel(r'$\epsilon_2 / K$', fontsize=13)
    ax.set_ylabel(r'$T_X$ (ms)', fontsize=13)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    all_data = {}
    for delta_over_K, color in zip(delta_list, colors):
        P_norm, T_X_vals, _, T_X_ms = _compute_T_X_vs_P(
            delta_over_K * g.K, kappa1_over_K, n_th,
            P_min_over_K, P_max_over_K, n_points, N_max, n_workers)
        ax.semilogy(P_norm, T_X_ms, '.-', color=color, markersize=4, lw=1.2,
                    label=rf'$\Delta/K={delta_over_K:.0f}$')
        all_data[delta_over_K] = (P_norm, T_X_vals, T_X_ms)

    ax.legend(fontsize=9, loc='upper left')
    _save_fig('figure_P_versus_Delta.png')
    return all_data


def plot_figure_Delta_versus_P(
    P_list        = (1.5, 2.17, 3.0, 4.0),  # P/K values — one curve each
    kappa1_over_K = None,                    # default: T₁ = 20 μs
    n_th          = 0.05,
    delta_min_K   = 0.0,
    delta_max_K   = 10.0,
    n_points      = 60,
    N_max         = 80,
    n_workers     = None,
):
    """
    T_X (ms) vs Δ/K, one semilogy curve per P value.
    X-axis: Δ/K.  Family parameter: P/K.
    Calls _compute_T_X_vs_delta (= plot_T_X_vs_delta computation) for each P.
    """
    if kappa1_over_K is None:
        kappa1_over_K = 1.0 / (g.K * 20e-6)   # T₁ = 20 μs → κ₁ = 1/T₁
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(P_list)))

    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        rf'$T_X$ vs $\Delta/K$ — $\kappa_1/K={kappa1_over_K:.2e}$, '
        rf'$n_{{th}}={n_th}$', fontsize=12)
    ax.set_xlabel(r'$\Delta / K$', fontsize=13)
    ax.set_ylabel(r'$T_X$ (ms)', fontsize=13)
    ax.set_xlim(-0.3, delta_max_K + 0.3)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    for m in range(0, int(delta_max_K) + 2, 2):
        ax.axvline(m, color='gray', ls='--', lw=0.8, alpha=0.4)

    all_data = {}
    for P_over_K, color in zip(P_list, colors):
        d_norm, T_X_vals, T_X_ms = _compute_T_X_vs_delta(
            P_over_K * g.K, kappa1_over_K, n_th,
            delta_min_K, delta_max_K, n_points, N_max, n_workers)
        ax.semilogy(d_norm, T_X_ms, '.-', color=color, markersize=4, lw=1.2,
                    label=rf'$\epsilon_2/K={P_over_K:.2f}$')
        all_data[P_over_K] = (d_norm, T_X_vals, T_X_ms)

    ax.legend(fontsize=9, loc='upper right')
    _save_fig('figure_Delta_versus_P.png')
    return all_data



if __name__ == "__main__":

    plot_figure_P_versus_Delta()
    plot_figure_Delta_versus_P()

   #plot_T_X_vs_P(delta_val=0.0, kappa1_over_K=1/50, P_max_over_K=12.0, n_points=50, N_max=60)
   #plot_T_X_vs_delta(P_val=2.17*g.K, kappa1_over_K=1e-3)