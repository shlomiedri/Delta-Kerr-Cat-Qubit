"""
extras/debug_tools.py
---------------------
Debugging utilities for inspecting the Husimi Q-function peak detection
at a single (P/K, Delta/K) operating point.

Not wired into the main CLI menu — run this file directly.

Usage
-----
    python extras/debug_tools.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from core.system import KerrCatSystem
from analysis.phase_space import PhaseSpaceDiagram


def debug_peak_detection(
    P_over_K:    float = 1.35,
    delta_over_K: float = 6.0,
    N:           int   = 40,
    n_th:        float = 0.05,
    grid:        int   = 60,
    limit:       float = 3.5,
    min_dist:    float = 0.8,
):
    """
    Plot the Husimi Q-function for a single point and overlay the detected peaks,
    so you can visually verify the peak-counting logic.

    Parameters
    ----------
    P_over_K / delta_over_K : operating point
    N       : Hilbert space size
    n_th    : thermal occupancy
    grid    : Q-function grid resolution
    limit   : phase-space window half-width
    min_dist: minimum separation between distinct peaks (in alpha units)
    """
    print(f"Debug point: P/K={P_over_K}, Delta/K={delta_over_K}, N={N}, n_th={n_th}")

    sys_  = KerrCatSystem(P_over_K, delta_over_K, N, n_th)
    rho   = sys_.steady_state
    Q, xvec, yvec = PhaseSpaceDiagram._husimi_q(rho, grid=grid, limit=limit)

    n_peaks = PhaseSpaceDiagram._count_peaks(Q, xvec, yvec, min_dist=min_dist)
    print(f"Detected peaks: {n_peaks}")

    # Rerun peak finding to extract candidate positions for plotting
    import scipy.ndimage as ndimage

    px_size = (xvec[-1] - xvec[0]) / (grid - 1)
    nbhd    = max(1, int(min_dist / px_size))

    data_max = ndimage.maximum_filter(Q, size=nbhd)
    maxima   = (Q == data_max) & (Q >= 0.1 * np.max(Q))
    labeled, n_obj = ndimage.label(maxima)
    raw = ndimage.center_of_mass(maxima, labeled, range(1, n_obj + 1))
    if not isinstance(raw, list):
        raw = [raw]

    peak_x = [xvec[0] + (xvec[-1] - xvec[0]) * (xi / (grid - 1)) for _, xi in raw]
    peak_y = [yvec[0] + (yvec[-1] - yvec[0]) * (yi / (grid - 1)) for yi, _ in raw]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    ax.contourf(xvec, yvec, Q, 80, cmap="inferno")
    ax.scatter(peak_x, peak_y, c="cyan", marker="x", s=120, lw=2,
               zorder=5, label=f"{len(raw)} raw candidates")
    ax.set_title(
        rf"Q-function  ($P/K={P_over_K}$, $\Delta/K={delta_over_K}$)"
        "\n"
        rf"Detected peaks after merging: {n_peaks}",
        fontsize=12,
    )
    ax.set_xlabel(r"Re($\alpha$)", fontsize=12)
    ax.set_ylabel(r"Im($\alpha$)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    debug_peak_detection(P_over_K=1.35, delta_over_K=6.0, N=40, n_th=0.05)
