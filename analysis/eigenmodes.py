"""
analysis/eigenmodes.py
----------------------
LiouvillianModes: computes and visualises the slowest eigenmodes of the
Liouvillian superoperator for a given KerrCatSystem operating point.

Physical interpretation of modes (three-solution regime)
---------------------------------------------------------
  Mode 0 : steady state  (Re(lambda) = 0)
  Mode 1 : bit-flip      |+alpha> <-> |-alpha>   [slowest decay]
  Mode 2 : vac leakage   cats -> |0>             [second slowest]
  Mode 3+ : fast intra-manifold relaxation

References
----------
  Mirrahimi et al., New J. Phys. 16, 045014 (2014)
  Grimm et al., Nature 584, 205 (2020)

Usage
-----
    from core.system import KerrCatSystem
    from analysis.eigenmodes import LiouvillianModes

    sys   = KerrCatSystem(P_over_K=1.35, delta_over_K=6.0, N=40, n_th=0.05)
    modes = LiouvillianModes(sys)
    modes.compute(n_modes=6)
    modes.print_eigenvalues()
    modes.plot_wigner(save_path="output/eigenmodes.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.sparse.linalg as spla
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core.system import KerrCatSystem


class LiouvillianModes:
    """
    Computes the slowest Liouvillian eigenmodes via sparse shift-invert
    around sigma=0, then visualises them as Wigner functions.

    Parameters
    ----------
    system : KerrCatSystem
    """

    # Human-readable labels for the first several modes
    _MODE_LABELS = [
        "steady state",
        "bit-flip (slow)",
        "vac leakage",
        "mode 3",
        "mode 4",
        "mode 5",
        "mode 6",
        "mode 7",
    ]

    def __init__(self, system: KerrCatSystem):
        self.system  = system
        self._evals  = None   # shape (n_modes,)
        self._evecs  = None   # shape (N^2, n_modes)

    # ── Public interface ─────────────────────────────────────────────────────

    def compute(self, n_modes: int = 6) -> None:
        """
        Compute the n_modes slowest Liouvillian eigenvalues / eigenmatrices.
        Results are sorted by |Re(lambda)| ascending (Mode 0 = steady state).
        """
        L     = self.system.liouvillian
        L_mat = L.data.as_scipy() if hasattr(L.data, "as_scipy") else L.data

        print(f"  Computing {n_modes} Liouvillian eigenmodes (N={self.system.N})...")
        evals, evecs = spla.eigs(
            L_mat, k=n_modes, sigma=0.0, return_eigenvectors=True
        )

        idx          = np.argsort(np.abs(np.real(evals)))
        self._evals  = evals[idx]
        self._evecs  = evecs[:, idx]
        print("  Done.")

    def print_eigenvalues(self) -> None:
        """Print a formatted table of all computed eigenvalues."""
        self._require_computed()
        N = self.system.N

        print(f"\n{'─'*60}")
        print(f"  Liouvillian eigenvalues  —  {self.system}")
        print(f"{'─'*60}")
        print(f"  {'Mode':<6}  {'Label':<20}  {'Re(lambda)':>12}  {'Tau (1/K)':>10}")
        print(f"  {'─'*6}  {'─'*20}  {'─'*12}  {'─'*10}")

        for i, ev in enumerate(self._evals):
            ri  = np.abs(np.real(ev))
            lbl = self._MODE_LABELS[i] if i < len(self._MODE_LABELS) else f"mode {i}"
            tau = f"{1/ri:.2f}" if ri > 1e-14 else "∞  (SS)"
            print(f"  {i:<6}  {lbl:<20}  {np.real(ev):>12.4e}  {tau:>10}")

        print(f"{'─'*60}\n")

    def plot_wigner(self, n_show: int = 5, save_path: str = None) -> None:
        """
        Plot Wigner functions for the first n_show eigenmodes in a 2×3 grid.
        Mode 0 is normalised to trace = 1 (it is the steady state).

        Parameters
        ----------
        n_show    : int — how many modes to plot (max 5 to fit the 2×3 grid)
        save_path : str or None
        """
        self._require_computed()

        n_show = min(n_show, len(self._evals), 5)
        N      = self.system.N
        xvec   = np.linspace(-4, 4, 100)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150,
                                 constrained_layout=True)
        axes_flat = axes.flatten()

        for i in range(n_show):
            vec   = self._evecs[:, i]
            mat   = vec.reshape((N, N), order="F")
            rho_i = qt.Qobj(mat)

            if i == 0:                          # steady state: normalise trace
                rho_i = rho_i / rho_i.tr()

            W     = qt.wigner(rho_i, xvec, xvec)
            ax    = axes_flat[i]
            limit = np.max(np.abs(W))

            im = ax.contourf(xvec, xvec, W, 100,
                             cmap="RdBu_r", vmin=-limit, vmax=limit)

            lbl = self._MODE_LABELS[i] if i < len(self._MODE_LABELS) else f"mode {i}"
            ax.set_title(
                rf"Mode {i}  [{lbl}]" + "\n"
                + rf"Re($\lambda$) = {np.real(self._evals[i]):.3e}",
                fontsize=13, pad=8,
            )
            ax.set_aspect("equal")
            if i >= 3:
                ax.set_xlabel(r"Re($\alpha$)", fontsize=12)
            if i % 3 == 0:
                ax.set_ylabel(r"Im($\alpha$)", fontsize=12)

            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="5%", pad=0.08)
            fig.colorbar(im, cax=cax)

        # Hide unused subplot(s)
        for j in range(n_show, 6):
            axes_flat[j].set_visible(False)

        sys = self.system
        fig.suptitle(
            rf"Liouvillian Eigenmodes  —  "
            rf"$P/K={sys.P_over_K}$,  $\Delta/K={sys.delta_over_K}$,  "
            rf"$n_{{th}}={sys.n_th}$,  $N={sys.N}$",
            fontsize=13, fontweight="bold",
        )

        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Saved: {save_path}")

        plt.show()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _require_computed(self) -> None:
        if self._evals is None:
            raise RuntimeError("Call compute() before accessing results.")
