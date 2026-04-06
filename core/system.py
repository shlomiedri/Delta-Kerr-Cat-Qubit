"""
core/system.py
--------------
KerrCatSystem: the physics core of the Delta-Kerr-Cat qubit simulation.

All Hamiltonian and collapse-operator terms are normalised by K so that
matrix elements are O(1) — identical convention to the original codebase.

Usage
-----
    from core.system import KerrCatSystem

    sys = KerrCatSystem(P_over_K=1.35, delta_over_K=6.0, N=40, n_th=0.05)
    H       = sys.hamiltonian        # QuTiP Qobj
    c_ops   = sys.collapse_ops       # list of QuTiP Qobj
    rho_ss  = sys.steady_state       # density matrix (computed on first access)
"""

import functools
import numpy as np
import qutip as qt

from core import params as p


class KerrCatSystem:
    """
    Encapsulates the physics of a single Kerr-cat qubit operating point.

    Parameters
    ----------
    P_over_K     : float  — two-photon drive strength in units of K
    delta_over_K : float  — cavity detuning in units of K
    N            : int    — Hilbert space truncation (Fock basis size)
    n_th         : float  — mean thermal photon number of the bath
    """

    def __init__(
        self,
        P_over_K:     float = 1.35,
        delta_over_K: float = 6.0,
        N:            int   = 40,
        n_th:         float = 0.05,
    ):
        self.P_over_K     = P_over_K
        self.delta_over_K = delta_over_K
        self.N            = N
        self.n_th         = n_th

    # ── Derived physical quantities ──────────────────────────────────────────

    @property
    def P_phys(self) -> float:
        """Two-photon drive in rad/s."""
        return self.P_over_K * p.K

    @property
    def delta_phys(self) -> float:
        """Detuning in rad/s."""
        return self.delta_over_K * p.K

    @property
    def cat_amplitude(self) -> float:
        """Cat-state amplitude alpha = sqrt(P/K)."""
        return np.sqrt(self.P_over_K)

    # ── Operators (rebuilt if N changes) ────────────────────────────────────

    @functools.cached_property
    def operators(self) -> dict:
        """
        Basic bosonic operators in the Fock basis of size N.
        Returns dict with keys: a, a_dag, n, x, p
        """
        a = qt.destroy(self.N)
        return {
            "a":     a,
            "a_dag": a.dag(),
            "n":     a.dag() * a,
            "x":     (a + a.dag()) / np.sqrt(2),
            "p":     1j * (a.dag() - a) / np.sqrt(2),
        }

    @functools.cached_property
    def hamiltonian(self) -> qt.Qobj:
        """
        Dimensionless Hamiltonian normalised by K:
            H/K = -(a†²a²) + (P/K)(a² + a†²) + (Delta/K)(a†a)

        References
        ----------
        Mirrahimi et al., New J. Phys. 16, 045014 (2014)
        Grimm et al., Nature 584, 205 (2020)
        """
        ops = self.operators
        kerr   = -(p.K / p.K)           * (ops["a_dag"] ** 2 * ops["a"] ** 2)
        pump   =  (self.P_phys / p.K)   * (ops["a"] ** 2 + ops["a_dag"] ** 2)
        detune =  (self.delta_phys / p.K) * ops["n"]
        return kerr + pump + detune

    @functools.cached_property
    def collapse_ops(self) -> list:
        """
        Single-photon loss and thermal excitation, both normalised by K:
            c_loss   = sqrt(kappa1*(1+n_th)/K) * a
            c_thermal= sqrt(kappa1*n_th/K)     * a†
        """
        a = self.operators["a"]
        rate = p.kappa_1 / p.K
        return [
            np.sqrt(rate * (1 + self.n_th)) * a,
            np.sqrt(rate * self.n_th)       * a.dag(),
        ]

    @functools.cached_property
    def liouvillian(self) -> qt.Qobj:
        """Full Liouvillian superoperator L[rho] = -i[H,rho] + dissipators."""
        return qt.liouvillian(self.hamiltonian, self.collapse_ops)

    @functools.cached_property
    def steady_state(self) -> qt.Qobj:
        """
        Steady-state density matrix rho_ss satisfying L[rho_ss] = 0.
        Computed once and cached — can take several seconds.
        """
        return qt.steadystate(self.hamiltonian, self.collapse_ops)

    # ── Convenience ─────────────────────────────────────────────────────────

    def coherent_state(self, alpha: complex) -> qt.Qobj:
        """Return density matrix for coherent state |alpha>."""
        return qt.ket2dm(qt.coherent(self.N, alpha))

    def __repr__(self) -> str:
        return (
            f"KerrCatSystem("
            f"P/K={self.P_over_K}, "
            f"Δ/K={self.delta_over_K}, "
            f"N={self.N}, "
            f"n_th={self.n_th}, "
            f"α={self.cat_amplitude:.3f})"
        )
