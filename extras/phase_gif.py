"""
extras/phase_gif.py
-------------------
Generates a GIF animation of the Husimi Q-function for fixed P/K
and a sweep of Delta/K values.

Not wired into the main CLI menu — run this file directly.

Usage
-----
    python extras/phase_gif.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import qutip as qt

from core import params as p
from core.system import KerrCatSystem


def make_phase_space_gif(
    P_over_K:    float = 3.5,
    delta_min:   float = -10.0,
    delta_max:   float =  10.0,
    delta_step:  float =   0.5,
    N:           int   =  60,
    n_th:        float =   0.05,
    fps:         int   =  10,
    save_name:   str   = "output/phase_space_sweep.gif",
):
    """
    Animate the steady-state Husimi Q-function while sweeping Delta/K.

    Parameters
    ----------
    P_over_K   : fixed drive strength
    delta_min/max/step : Delta/K sweep range
    N          : Hilbert space size
    n_th       : thermal occupancy
    fps        : animation frame rate
    save_name  : output GIF path
    """
    deltas = np.arange(delta_min, delta_max + 1e-9, delta_step)
    xvec   = np.linspace(-3.5, 3.5, 80)
    yvec   = np.linspace(-3.5, 3.5, 80)

    print(f"Generating GIF: P/K={P_over_K}, Delta/K in [{delta_min}, {delta_max}]")
    print(f"Frames: {len(deltas)},  N={N},  fps={fps}")

    fig, ax = plt.subplots(figsize=(6, 6))
    dummy   = np.zeros((len(xvec), len(yvec)))
    contour = ax.contourf(xvec, yvec, dummy, 100, cmap="inferno")
    cbar    = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$Q(\alpha)$", fontsize=12)

    def update(frame_idx):
        ax.clear()
        d = round(deltas[frame_idx], 2)

        sys_ = KerrCatSystem(P_over_K, d, N, n_th)
        Q    = qt.qfunc(sys_.steady_state, xvec, yvec)

        v_max = max(np.max(Q), 1e-4)
        ax.contourf(xvec, yvec, Q, 100, cmap="inferno", vmin=0, vmax=v_max)
        ax.set_title(
            rf"$P/K={P_over_K}$,  $\Delta/K={d:.2f}$"
            "\n"
            rf"($n_{{th}}={n_th}$,  $\kappa_1/K={p.kappa_1/p.K:.2e}$)"
        )
        ax.set_xlabel(r"Re($\alpha$)")
        ax.set_ylabel(r"Im($\alpha$)")
        ax.set_aspect("equal")
        ax.axhline(0, color="white", alpha=0.3, ls="--", lw=1)
        ax.axvline(0, color="white", alpha=0.3, ls="--", lw=1)
        print(f"  Frame {frame_idx+1}/{len(deltas)}  Delta/K={d:.2f}", end="\r")
        return []

    anim = FuncAnimation(fig, update, frames=len(deltas), interval=100, blit=False)
    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)
    anim.save(save_name, writer=PillowWriter(fps=fps))
    print(f"\nSaved: {save_name}")


if __name__ == "__main__":
    make_phase_space_gif()
