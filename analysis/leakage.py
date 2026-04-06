"""
analysis/leakage.py
-------------------
LeakageAnalysis: demonstrates the two-timescale relaxation structure of the
Delta-Kerr-Cat qubit by comparing two initial states.

Physical motivation
-------------------
In the three-solution regime the Liouvillian has (at least) two slow modes:

  Mode 1 — bit-flip    |+alpha> <-> |-alpha>       (slowest,  tau_1)
  Mode 2 — vac leakage cats -> vacuum              (faster,   tau_2)

The trace distance D(rho(t), rho_ss) decomposes as:
  D(t) ~ A1*exp(-|lambda1|*t) + A2*exp(-|lambda2|*t) + faster modes ...

Key insight
-----------
  rho_left = |-alpha><-alpha|
      Maximally left-right asymmetric => large A1, small A2.
      Late-time fit matches Mode 1 (bit-flip).

  rho_sym = (|+alpha><+alpha| + |-alpha><-alpha|) / 2
      Symmetric => A1 = 0 exactly.
      Late-time fit matches Mode 2 (vacuum leakage).   [Ref 1, 2]

References
----------
  [1] Mirrahimi et al., New J. Phys. 16, 045014 (2014)
  [2] Grimm et al., Nature 584, 205 (2020)
  [3] Puri et al., npj Quantum Inf. 3, 18 (2017)
  [4] Drummond & Walls, J. Phys. A 13, 725 (1980)

Usage
-----
    from core.system import KerrCatSystem
    from analysis.leakage import LeakageAnalysis

    sys  = KerrCatSystem(P_over_K=1.35, delta_over_K=6.0, N=40, n_th=0.05)
    leak = LeakageAnalysis(sys)
    leak.run()              # mesolve both initial states
    leak.fit()              # exponential fits on late window
    leak.print_summary()    # comparison table
    leak.plot(save_path="output/leakage.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.sparse.linalg as spla
from tqdm import tqdm

from core.system import KerrCatSystem


class LeakageAnalysis:
    """
    Runs time evolution from two initial states and extracts Mode 1 / Mode 2
    decay rates by fitting exponentials to the late-time trace distance.

    Parameters
    ----------
    system : KerrCatSystem
    """

    def __init__(self, system: KerrCatSystem):
        self.system = system

        # Results filled by run() and fit()
        self._tlist     = None
        self._td_left   = None   # trace distance for rho_left
        self._td_sym    = None   # trace distance for rho_sym
        self._vac_left  = None   # vacuum population for rho_left
        self._vac_sym   = None   # vacuum population for rho_sym
        self._evals     = None   # Liouvillian eigenvalues

        self._gamma_left = None  # fitted decay rate (left state)
        self._tau_left   = None
        self._yfit_left  = None

        self._gamma_sym  = None  # fitted decay rate (sym state)
        self._tau_sym    = None
        self._yfit_sym   = None

    # ── Public interface ─────────────────────────────────────────────────────

    def run(self, T_max_factor: int = 8, n_steps: int = 500) -> None:
        """
        Compute Liouvillian eigenvalues, then solve the master equation for
        both rho_left and rho_sym.

        Parameters
        ----------
        T_max_factor : T_max = T_max_factor / |Re(lambda_2)|
        n_steps      : number of time steps in mesolve
        """
        sys = self.system

        # Step 1: Liouvillian eigenvalues (need tau_2 to set T_max)
        print(f"\n  Computing Liouvillian eigenvalues...")
        self._evals = self._get_eigenvalues(n_modes=6)
        rate2 = np.abs(np.real(self._evals[2]))
        tau2  = 1.0 / rate2
        rate1 = np.abs(np.real(self._evals[1]))
        print(f"  Mode 1 (bit-flip)    : rate={rate1:.4e}  tau={1/rate1:.1f}")
        print(f"  Mode 2 (vac leakage) : rate={rate2:.4e}  tau={tau2:.1f}")

        # Step 2: Steady state
        print("  Computing steady state...", end=" ", flush=True)
        rho_ss = sys.steady_state
        print("done.")

        # Step 3: Build initial states
        alpha     = sys.cat_amplitude
        rho_left  = sys.coherent_state(-alpha)
        rho_sym   = 0.5 * sys.coherent_state(alpha) + 0.5 * sys.coherent_state(-alpha)

        # Step 4: Time evolution
        T_max = T_max_factor * tau2
        print(f"\n  T_max = {T_max_factor} × tau2 = {T_max:.1f}  (units of 1/K)")

        print("  Evolving rho_left ...")
        tlist, td_l, vac_l = self._mesolve(rho_left, rho_ss, T_max, n_steps)

        print("  Evolving rho_sym ...")
        _,     td_s, vac_s = self._mesolve(rho_sym,  rho_ss, T_max, n_steps)

        self._tlist    = tlist
        self._td_left  = td_l
        self._td_sym   = td_s
        self._vac_left = vac_l
        self._vac_sym  = vac_s

    def fit(self, late_start_frac: float = 0.3) -> None:
        """
        Fit a single exponential to the late-time trace distance for each state.

        Parameters
        ----------
        late_start_frac : float — fit window starts at this fraction of T_max
        """
        self._require_run()

        t_start = self._tlist[0] + late_start_frac * (self._tlist[-1] - self._tlist[0])
        t_end   = self._tlist[-1]

        self._gamma_left, self._tau_left, self._yfit_left = self._fit_exp(
            self._tlist, self._td_left, t_start, t_end, label="left"
        )
        self._gamma_sym, self._tau_sym, self._yfit_sym = self._fit_exp(
            self._tlist, self._td_sym,  t_start, t_end, label="sym"
        )

    def print_summary(self) -> None:
        """Print comparison table: fit rates vs Liouvillian eigenvalues."""
        self._require_run()
        if self._gamma_left is None:
            raise RuntimeError("Call fit() before print_summary().")

        rate1 = np.abs(np.real(self._evals[1]))
        rate2 = np.abs(np.real(self._evals[2]))

        def row(label, rate, tau, ref_rate):
            ratio = rate / ref_rate if ref_rate > 0 else float("nan")
            return (f"  {label:<40}  {rate:>12.4e}  {tau:>10.2f}  "
                    f"ratio={ratio:.3f}")

        print(f"\n{'═'*75}")
        print(f"  LEAKAGE ANALYSIS  —  {self.system}")
        print(f"{'═'*75}")
        print(f"  {'Source':<40}  {'Rate':>12}  {'Tau (1/K)':>10}")
        print(f"  {'─'*40}  {'─'*12}  {'─'*10}")

        labels = ["steady state", "bit-flip", "vac leakage",
                  "mode 3", "mode 4", "mode 5"]
        for i, ev in enumerate(self._evals):
            ri  = np.abs(np.real(ev))
            lbl = labels[i] if i < len(labels) else f"mode {i}"
            if ri < 1e-14:
                print(f"  Louv Mode {i} ({lbl:<14})  rate ~ 0  (steady state)")
            else:
                print(f"  Louv Mode {i} ({lbl:<14})  {ri:>12.4e}  {1/ri:>10.2f}")

        print()
        if self._gamma_left is not None:
            print(row("Left fit  [late]  vs Mode 1 (bit-flip)",
                      self._gamma_left, self._tau_left, rate1))
        if self._gamma_sym is not None:
            print(row("Sym  fit  [late]  vs Mode 2 (vac leakage)",
                      self._gamma_sym, self._tau_sym, rate2))

        print(f"\n  ratio ~ 1.0  =>  fit matches Liouvillian eigenvalue")
        print(f"{'═'*75}\n")

    def plot(self, save_path: str = None) -> None:
        """
        Three-panel plot:
          1. Log-scale trace distance for both states + eigenvalue references
          2. Zoom on t <= 3*tau2 (linear)
          3. Vacuum population for both states

        Parameters
        ----------
        save_path : str or None
        """
        self._require_run()

        rate1 = np.abs(np.real(self._evals[1]))
        rate2 = np.abs(np.real(self._evals[2]))
        tau2  = 1.0 / rate2
        sys   = self.system

        a    = qt.destroy(sys.N)
        ss_v = qt.expect(qt.fock_dm(sys.N, 0), sys.steady_state)

        fig, axes = plt.subplots(3, 1, figsize=(11, 13), dpi=130)
        fig.suptitle(
            rf"Vacuum Leakage Analysis  —  "
            rf"$P/K={sys.P_over_K}$,  $\Delta/K={sys.delta_over_K}$,  "
            rf"$n_{{th}}={sys.n_th}$,  $N={sys.N}$",
            fontsize=13, fontweight="bold",
        )

        # ── Panel 1: log-scale trace distance ────────────────────────────────
        ax = axes[0]
        v_l = self._td_left > 1e-12
        v_s = self._td_sym  > 1e-12
        ref_l = self._td_left[v_l][0]
        ref_s = self._td_sym[v_s][0]

        ax.semilogy(self._tlist[v_l], self._td_left[v_l], "k-",  lw=1.8,
                    label=r"$\rho_{left} = |-\alpha\rangle\langle-\alpha|$")
        ax.semilogy(self._tlist[v_s], self._td_sym[v_s],  color="darkorange", lw=1.8,
                    label=r"$\rho_{sym}  = \frac{1}{2}(|{+}\alpha\rangle\langle{+}\alpha|"
                          r" + |-\alpha\rangle\langle-\alpha|)$")

        # Liouvillian reference slopes
        t = self._tlist
        ax.semilogy(t, ref_l * np.exp(-rate1 * t), ":", color="royalblue", lw=1.5,
                    label=rf"Mode 1 (bit-flip): $\tau_1={1/rate1:.1f}$")
        ax.semilogy(t, ref_s * np.exp(-rate2 * t), ":", color="firebrick", lw=1.5,
                    label=rf"Mode 2 (vac leakage): $\tau_2={tau2:.1f}$")

        # Fit overlays
        if self._yfit_left is not None:
            ax.semilogy(t, np.abs(self._yfit_left), "--", color="royalblue", lw=2,
                        label=rf"Left fit: $\tau={self._tau_left:.1f}$  "
                              rf"(ratio={self._gamma_left/rate1:.3f})")
        if self._yfit_sym is not None:
            ax.semilogy(t, np.abs(self._yfit_sym),  "--", color="firebrick",  lw=2,
                        label=rf"Sym fit:  $\tau={self._tau_sym:.1f}$  "
                              rf"(ratio={self._gamma_sym/rate2:.3f})")

        # Fit window shade
        t_late = t[0] + 0.3 * (t[-1] - t[0])
        ax.axvspan(t_late, t[-1], alpha=0.05, color="gray", label="Fit window")

        ax.set_ylabel(r"Trace distance $D$ (log)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # ── Panel 2: linear zoom to 3*tau2 ───────────────────────────────────
        ax       = axes[1]
        zoom_end = min(3 * tau2, t[-1])
        zm       = t <= zoom_end

        ax.plot(t[zm], self._td_left[zm], "k-",          lw=1.8,
                label=r"$\rho_{left}$")
        ax.plot(t[zm], self._td_sym[zm],  color="darkorange", lw=1.8,
                label=r"$\rho_{sym}$")
        if self._yfit_left is not None:
            ax.plot(t[zm], np.abs(self._yfit_left[zm]), "--", color="royalblue", lw=1.5,
                    label=rf"Left fit $\tau={self._tau_left:.1f}$")
        if self._yfit_sym is not None:
            ax.plot(t[zm], np.abs(self._yfit_sym[zm]),  "--", color="firebrick",  lw=1.5,
                    label=rf"Sym fit  $\tau={self._tau_sym:.1f}$")

        ax.set_xlim(0, zoom_end)
        ax.set_ylabel(rf"Trace distance (zoom $t \leq {zoom_end:.0f}$)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 3: vacuum population ────────────────────────────────────────
        ax = axes[2]
        ax.plot(t, self._vac_left, "k-",          lw=1.5,
                label=r"$\langle 0|\rho_{left}(t)|0\rangle$")
        ax.plot(t, self._vac_sym,  color="darkorange", lw=1.5,
                label=r"$\langle 0|\rho_{sym}(t)|0\rangle$")
        ax.axhline(ss_v, color="red", ls="--", lw=1.2, alpha=0.7,
                   label=rf"SS $\langle 0|\rho_{{ss}}|0\rangle = {ss_v:.3f}$")
        ax.set_xlabel(r"Time (units of $1/K$)", fontsize=11)
        ax.set_ylabel(r"Vacuum population", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Saved: {save_path}")

        plt.show()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_eigenvalues(self, n_modes: int = 6) -> np.ndarray:
        L     = self.system.liouvillian
        L_mat = L.data.as_scipy() if hasattr(L.data, "as_scipy") else L.data
        evals, _ = spla.eigs(L_mat, k=n_modes, sigma=0.0, return_eigenvectors=True)
        return evals[np.argsort(np.abs(np.real(evals)))]

    def _mesolve(self, rho0, rho_ss, T_max, n_steps):
        """Run mesolve and return (tlist, trace_dist, vac_pop)."""
        sys   = self.system
        a     = qt.destroy(sys.N)
        tlist = np.linspace(0, T_max, n_steps)
        opts  = {
            "store_states": True,
            "nsteps": 2_000_000,
            "method": "bdf",
            "atol": 1e-8,
            "rtol": 1e-6,
            "progress_bar": True,
        }
        result = qt.mesolve(
            sys.hamiltonian, rho0, tlist, sys.collapse_ops,
            e_ops=[qt.fock_dm(sys.N, 0)], options=opts,
        )
        trace_dist = np.array([
            qt.tracedist(s, rho_ss)
            for s in tqdm(result.states, desc="    Trace dist", leave=False)
        ])
        return tlist, trace_dist, np.array(result.expect[0])

    @staticmethod
    def _fit_exp(t, y, t_start, t_end, eps=1e-12, label=""):
        """Fit y ~ A*exp(-gamma*t) on [t_start, t_end]. Returns (gamma, tau, y_fit)."""
        mask = (t >= t_start) & (t <= t_end) & (y > eps)
        if np.sum(mask) < 2:
            print(f"  [{label}] Not enough points for fit.")
            return None, None, None
        coeffs = np.polyfit(t[mask], np.log(y[mask]), 1)
        gamma  = -coeffs[0]
        A      = np.exp(coeffs[1])
        if gamma <= 0:
            print(f"  [{label}] Non-positive decay rate.")
            return None, None, None
        return gamma, 1.0 / gamma, A * np.exp(-gamma * t)

    def _require_run(self):
        if self._tlist is None:
            raise RuntimeError("Call run() before this method.")
