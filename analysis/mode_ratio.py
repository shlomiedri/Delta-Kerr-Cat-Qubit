"""
analysis/mode_ratio.py
----------------------
ModeRatioAnalysis: sweeps the (P/K, Delta/K) plane and plots
log10(tau_bitflip / tau_leakage) restricted to the 3-solution region.

Two-pass algorithm
------------------
  Pass 1  (fast, parallel) : Q-function peak count → find 3-solution cells
  Pass 2  (slow, parallel) : Liouvillian eigenvalues at 3-solution cells only

Mode identification
-------------------
  For each eigenmatrix R, the discriminator is:

      asymmetry = | Tr[R · a] |   (complex magnitude)

  Bit-flip mode  (|+α⟩⟨+α| − |−α⟩⟨−α|) :  asymmetry = 2|α|
  Leakage mode   (|+α⟩⟨+α| + |−α⟩⟨−α| − c|0⟩⟨0|) :  asymmetry = 0

  Rationale: ⟨α|a|α⟩ = α  and  ⟨−α|a|−α⟩ = −α.
  For the antisymmetric (bit-flip) combination Tr[R·a] = α − (−α) = 2α ≠ 0.
  For the symmetric (leakage) combination Tr[R·a] = α + (−α) = 0.
  Complex magnitude (not Re) is used to be invariant to the arbitrary
  phase e^{iφ} that sparse eigensolvers attach to eigenvectors.

Heatmap
-------
  Color    : log10(τ_bitflip / τ_leakage)   — diverging RdBu_r colormap
  Gray     : non-3-solution cells            — cmap.set_bad('lightgray')
  Hatching : cells where leakage is slower   — diagonal '///' overlay
  Contour  : bold black line at log = 0      — equal timescale boundary

Usage
-----
    from analysis.mode_ratio import ModeRatioAnalysis

    mr = ModeRatioAnalysis(N=40, n_th=0.05,
                           P_range=(0.5, 3.0), delta_range=(0.0, 15.0),
                           P_step=0.1, delta_step=0.3)
    mr.run()
    mr.plot(save_path="output/mode_ratio.png")
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import qutip as qt
import scipy.sparse.linalg as spla

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from analysis.phase_space import PhaseSpaceDiagram
from core.system import KerrCatSystem


# ─────────────────────────────────────────────────────────────────────────────
# Module-level workers (must be picklable for Windows multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _scan_pixel(args):
    """Pass 1 worker: return Q-function peak count for one (P/K, Delta/K) point."""
    P_over_K, delta_over_K, N, n_th = args
    sys_  = KerrCatSystem(P_over_K, delta_over_K, N, n_th)
    rho   = sys_.steady_state
    Q, xvec, yvec = PhaseSpaceDiagram._husimi_q(rho)
    return PhaseSpaceDiagram._count_peaks(Q, xvec, yvec)


def _ratio_pixel(args):
    """
    Pass 2 worker: compute log10(tau_bitflip / tau_leakage) for one 3-solution point.

    Returns
    -------
    (log_ratio : float, inverted : bool)
        log_ratio = NaN if computation fails
        inverted  = True when leakage is slower than bit-flip
    """
    P_over_K, delta_over_K, N, n_th = args
    try:
        sys_  = KerrCatSystem(P_over_K, delta_over_K, N, n_th)
        L     = sys_.liouvillian
        L_mat = L.data.as_scipy() if hasattr(L.data, "as_scipy") else L.data
        a_op  = sys_.operators["a"]

        evals, evecs = spla.eigs(L_mat, k=3, sigma=0.0, return_eigenvectors=True)

        # Sort by |Re(lambda)| ascending — Mode 0 = steady state (Re ≈ 0)
        idx   = np.argsort(np.abs(np.real(evals)))
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Analyse Modes 1 and 2 (skip Mode 0)
        candidates = []
        for i in (1, 2):
            vec = evecs[:, i]
            R   = qt.Qobj(vec.reshape((N, N), order="F"),
                          dims=[[N], [N]])
            asymmetry = float(abs((R * a_op).tr()))
            rate      = float(abs(np.real(evals[i])))
            tau       = 1.0 / rate if rate > 1e-14 else np.inf
            candidates.append({"asymmetry": asymmetry, "tau": tau})

        # Higher asymmetry score → bit-flip mode
        if candidates[0]["asymmetry"] >= candidates[1]["asymmetry"]:
            tau_bf, tau_leak = candidates[0]["tau"], candidates[1]["tau"]
        else:
            tau_bf, tau_leak = candidates[1]["tau"], candidates[0]["tau"]

        if not np.isfinite(tau_bf) or not np.isfinite(tau_leak) or tau_leak == 0:
            return (np.nan, False)

        log_ratio = float(np.log10(tau_bf / tau_leak))
        inverted  = tau_leak > tau_bf          # leakage slower → unusual regime

        return (log_ratio, inverted)

    except Exception:
        return (np.nan, False)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class ModeRatioAnalysis:
    """
    Two-pass sweep: first identify 3-solution cells, then compute mode ratios.

    Parameters
    ----------
    N           : int   — Hilbert space truncation (shared with system params)
    n_th        : float — thermal photon occupancy
    P_range     : (float, float) — (min, max) P/K sweep range
    delta_range : (float, float) — (min, max) Delta/K sweep range
    P_step      : float — step in P/K direction
    delta_step  : float — step in Delta/K direction
    """

    def __init__(
        self,
        N:           int   = 40,
        n_th:        float = 0.05,
        P_range:     tuple = (0.5,  3.0),
        delta_range: tuple = (0.0, 15.0),
        P_step:      float = 0.1,
        delta_step:  float = 0.3,
    ):
        self.N           = N
        self.n_th        = n_th
        self.P_range     = P_range
        self.delta_range = delta_range
        self.P_step      = P_step
        self.delta_step  = delta_step

        self._p_vals      = None
        self._delta_vals  = None
        self._peaks_grid  = None   # int array — Q-function peak counts
        self._ratio_grid  = None   # float array — log10(tau_bf / tau_leak), NaN elsewhere
        self._inv_grid    = None   # bool array — True where mode ordering inverted

    # ── Public interface ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute both passes and store results."""
        self._p_vals     = np.arange(
            self.P_range[0],     self.P_range[1]     + 1e-9, self.P_step)
        self._delta_vals = np.arange(
            self.delta_range[0], self.delta_range[1] + 1e-9, self.delta_step)

        nP = len(self._p_vals)
        nD = len(self._delta_vals)
        n_workers = max(1, cpu_count() - 1)

        # ── Pass 1: count Q-function peaks (full grid) ───────────────────────
        tasks_p1 = [
            (float(P), float(d), self.N, self.n_th)
            for P in self._p_vals
            for d in self._delta_vals
        ]

        print(f"\nPass 1 — phase scan: {len(tasks_p1)} points  "
              f"(N={self.N}, n_th={self.n_th}, workers={n_workers})")
        t0 = time.time()
        with Pool(processes=n_workers) as pool:
            peaks_flat = list(tqdm(
                pool.imap(_scan_pixel, tasks_p1),
                total=len(tasks_p1),
                desc="Phase scan",
            ))
        print(f"  Done in {time.time() - t0:.1f}s")

        self._peaks_grid = np.array(peaks_flat, dtype=int).reshape(nP, nD)

        # ── Pass 2: eigenvalues at 3-solution cells only ─────────────────────
        three_sol_idx = [
            (i, j)
            for i in range(nP)
            for j in range(nD)
            if self._peaks_grid[i, j] == 3
        ]

        tasks_p2 = [
            (float(self._p_vals[i]), float(self._delta_vals[j]), self.N, self.n_th)
            for (i, j) in three_sol_idx
        ]

        print(f"\nPass 2 — eigenvalue sweep: {len(tasks_p2)} 3-solution points")
        t0 = time.time()
        with Pool(processes=n_workers) as pool:
            ratio_results = list(tqdm(
                pool.imap(_ratio_pixel, tasks_p2),
                total=len(tasks_p2),
                desc="Mode ratios",
            ))
        print(f"  Done in {time.time() - t0:.1f}s")

        # Build result grids (NaN / False for non-3-solution cells)
        self._ratio_grid = np.full((nP, nD), np.nan)
        self._inv_grid   = np.zeros((nP, nD), dtype=bool)

        for (i, j), (log_r, inv) in zip(three_sol_idx, ratio_results):
            self._ratio_grid[i, j] = log_r
            self._inv_grid[i, j]   = inv

    def plot(self, save_path: str = None) -> None:
        """
        Plot the mode-ratio heatmap.  Call run() first.

        Visual encoding
        ---------------
        Color      : log10(τ_bitflip / τ_leakage) — diverging RdBu_r
        Gray       : non-3-solution cells
        Hatching   : inverted cells (leakage slower than bit-flip)
        Contour    : bold black line where log10 ratio = 0
        """
        if self._ratio_grid is None:
            raise RuntimeError("Call run() before plot().")

        # Masked array: NaN cells become gray
        masked = np.ma.masked_invalid(self._ratio_grid)

        cmap = plt.cm.RdBu.copy()
        cmap.set_bad(color="lightgray")

        # Symmetric colorscale around 0
        finite_vals = self._ratio_grid[np.isfinite(self._ratio_grid)]
        if len(finite_vals) == 0:
            print("Warning: no finite ratio values found — nothing to plot.")
            return
        vlim = max(0.5, np.nanpercentile(np.abs(finite_vals), 98))

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        extent = [
            self._delta_vals[0] - self.delta_step / 2,
            self._delta_vals[-1] + self.delta_step / 2,
            self._p_vals[0]     - self.P_step / 2,
            self._p_vals[-1]    + self.P_step / 2,
        ]

        im = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=-vlim,
            vmax=+vlim,
            aspect="auto",
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r"$\log_{10}(\tau_\mathrm{bitflip}\ /\ \tau_\mathrm{leakage})$",
                       fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel(r"Detuning $\Delta/K$",            fontsize=14, fontweight="bold")
        ax.set_ylabel(r"Two-Photon Drive $\epsilon_2/K$", fontsize=14, fontweight="bold")
        ax.set_title(
            "Mode Ratio Heatmap — 3-Solution Region\n"
            rf"$N={self.N}$,  $n_{{th}}={self.n_th}$  |  "
            r"Red: bit-flip $\gg$ leakage     Blue: leakage $\gg$ bit-flip",
            fontsize=11, pad=10,
        )
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Saved: {save_path}")

        plt.show()
