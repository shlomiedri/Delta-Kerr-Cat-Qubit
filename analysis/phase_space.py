"""
analysis/phase_space.py
-----------------------
PhaseSpaceDiagram: generates and plots the steady-state phase diagram
of the Delta-Kerr-Cat qubit by counting Husimi Q-function peaks across
a 2D grid of (P/K, Delta/K) values.

Phase regions
-------------
  1 peak  → Vacuum only          (blue)
  2 peaks → Bistable cat states  (teal)
  3 peaks → Cat + vacuum coexistence  (orange/red)

Theory separatrix: P/K = |Delta/K| / 2

References
----------
  Mirrahimi et al., New J. Phys. 16, 045014 (2014)
  Drummond & Walls, J. Phys. A 13, 725 (1980)

Usage
-----
    from analysis.phase_space import PhaseSpaceDiagram

    diag = PhaseSpaceDiagram(N=60, n_th=0.05)
    diag.generate()   # parallel sweep (slow)
    diag.plot(save_path="output/phase_diagram.png")
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.ndimage as ndimage
import qutip as qt

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm

from core import params as p
from core.system import KerrCatSystem


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker (must be picklable for multiprocessing on Windows)
# ─────────────────────────────────────────────────────────────────────────────
def _process_pixel(args):
    """
    Compute the number of steady-state peaks for one (P/K, Delta/K) point.
    args = (P_over_K, delta_over_K, N, n_th)
    """
    P_over_K, delta_over_K, N, n_th = args
    sys   = KerrCatSystem(P_over_K, delta_over_K, N, n_th)
    rho   = sys.steady_state
    Q, xvec, yvec = PhaseSpaceDiagram._husimi_q(rho)
    return PhaseSpaceDiagram._count_peaks(Q, xvec, yvec)


class PhaseSpaceDiagram:
    """
    Sweeps (P/K, Delta/K) space and counts the number of Husimi Q-function
    peaks in the steady state, producing a three-colour phase diagram.

    Parameters
    ----------
    N           : int   — Hilbert space truncation
    n_th        : float — thermal photon occupancy
    P_range     : (float, float) — (min, max) of P/K sweep
    delta_range : (float, float) — (min, max) of Delta/K sweep
    P_step      : float — step size in P/K
    delta_step  : float — step size in Delta/K
    """

    def __init__(
        self,
        N:           int   = 60,
        n_th:        float = 0.05,
        P_range:     tuple = (0.0,  4.0),
        delta_range: tuple = (-10.0, 10.0),
        P_step:      float = 0.02,
        delta_step:  float = 0.1,
    ):
        self.N           = N
        self.n_th        = n_th
        self.P_range     = P_range
        self.delta_range = delta_range
        self.P_step      = P_step
        self.delta_step  = delta_step

        self._grid_data  = None   # filled by generate()
        self._p_vals     = None
        self._delta_vals = None

    # ── Public interface ─────────────────────────────────────────────────────

    def generate(self) -> None:
        """
        Run the parallel sweep over the (P/K, Delta/K) grid.
        Results are stored internally for plotting.
        """
        self._p_vals     = np.arange(self.P_range[0],     self.P_range[1]     + 1e-9, self.P_step)
        self._delta_vals = np.arange(self.delta_range[0], self.delta_range[1] + 1e-9, self.delta_step)

        tasks = [
            (P, d, self.N, self.n_th)
            for P in self._p_vals
            for d in self._delta_vals
        ]

        n_workers = max(1, cpu_count() - 1)
        print(f"\nPhase diagram sweep: {len(tasks)} points  "
              f"(N={self.N}, n_th={self.n_th}, workers={n_workers})")

        t0 = time.time()
        with Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_pixel, tasks),
                total=len(tasks),
                desc="Phase diagram",
            ))
        print(f"Done in {time.time() - t0:.1f}s")

        self._grid_data = np.array(results).reshape(
            len(self._p_vals), len(self._delta_vals)
        )

    def plot(self, save_path: str = None) -> None:
        """
        Plot the phase diagram.  Call generate() first.

        Parameters
        ----------
        save_path : str or None — if given, save PNG to this path
        """
        if self._grid_data is None:
            raise RuntimeError("Call generate() before plot().")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Colour map: 1=blue(vac), 2=teal(cats), 3=orange(coexistence)
        cmap  = ListedColormap(["#f0f0f5", "#7bb3ba", "#e07a5f"])
        norm  = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
        extent = [
            self._delta_vals[0], self._delta_vals[-1],
            self._p_vals[0],     self._p_vals[-1],
        ]

        ax.imshow(
            self._grid_data, origin="lower", extent=extent,
            cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
        )

        # Theory separatrix:  P/K = |Delta/K| / 2
        x_th = np.linspace(self._delta_vals[0], self._delta_vals[-1], 400)
        ax.plot(x_th, 0.5 * np.abs(x_th), "k--", lw=1.5, alpha=0.6,
                label=r"Theory: $\epsilon_2 = |\Delta|/2$")

        # Axes
        ax.set_xlabel(r"Detuning $\Delta/K$",           fontsize=14, fontweight="bold")
        ax.set_ylabel(r"Two-Photon Drive $\epsilon_2/K$", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Kerr-Cat Phase Diagram\n"
            rf"$N={self.N}$,  $\kappa_1/K={p.kappa_1/p.K:.2e}$,  $n_{{th}}={self.n_th:.2f}$",
            fontsize=14, pad=12,
        )
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.set_xlim(self._delta_vals[0], self._delta_vals[-1])
        ax.set_ylim(self._p_vals[0],     self._p_vals[-1])

        # Colour bar
        im   = ax.images[0]
        cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3], pad=0.02)
        cbar.ax.set_yticklabels(["1: Vacuum", "2: Cat states", "3: Coexistence"], fontsize=11)
        cbar.ax.tick_params(length=0)

        ax.legend(loc="upper center", fontsize=10, framealpha=0.9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")

        plt.show()

    # ── Static helpers (also used by _process_pixel) ─────────────────────────

    @staticmethod
    def _husimi_q(rho: qt.Qobj, grid: int = 60, limit: float = 3.5):
        """Compute Husimi Q-function on a square grid."""
        xvec = np.linspace(-limit, limit, grid)
        yvec = np.linspace(-limit, limit, grid)
        Q    = qt.qfunc(rho, xvec, yvec)
        return Q, xvec, yvec

    @staticmethod
    def _count_peaks(Q, xvec, yvec, min_dist: float = 0.8) -> int:
        """
        Count the number of physically distinct peaks in the Husimi Q-function.

        Algorithm
        ---------
        1. Find local maxima via a maximum filter (size = min_dist in alpha units)
        2. Filter out noise below 10% of global max
        3. Merge peaks closer than min_dist into one
        4. Classify peaks as vacuum (|alpha| < 0.5) or side peaks
        5. Enforce physical max of 2 side peaks (left + right cat)

        Returns
        -------
        int : 1 (vacuum), 2 (bistable cats), or 3 (coexistence)
        """
        grid       = Q.shape[0]
        px_size    = (xvec[-1] - xvec[0]) / (grid - 1)
        nbhd       = max(1, int(min_dist / px_size))

        data_max   = ndimage.maximum_filter(Q, size=nbhd)
        maxima     = Q == data_max
        maxima[Q < 0.1 * np.max(Q)] = False

        labeled, n_obj = ndimage.label(maxima)
        if n_obj == 0:
            return 0

        raw = ndimage.center_of_mass(maxima, labeled, range(1, n_obj + 1))
        if not isinstance(raw, list):
            raw = [raw]

        # Convert pixel indices → alpha coordinates
        candidates = []
        for y_idx, x_idx in raw:
            ax = xvec[0] + (xvec[-1] - xvec[0]) * (x_idx / (grid - 1))
            ay = yvec[0] + (yvec[-1] - yvec[0]) * (y_idx / (grid - 1))
            h  = Q[int(round(y_idx)), int(round(x_idx))]
            candidates.append({"pos": complex(ax, ay), "h": h})

        # Merge nearby peaks (keep tallest)
        candidates.sort(key=lambda c: c["h"], reverse=True)
        merged = []
        for cand in candidates:
            if all(abs(cand["pos"] - kept["pos"]) >= min_dist for kept in merged):
                merged.append(cand)

        # Classify
        has_vacuum = any(abs(c["pos"]) < 0.5 for c in merged)
        side_peaks = min(sum(abs(c["pos"]) >= 0.5 for c in merged), 2)

        return (1 if has_vacuum else 0) + side_peaks
