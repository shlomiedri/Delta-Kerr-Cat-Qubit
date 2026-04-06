"""
main.py
-------
Interactive CLI entry point for the Delta-Kerr-Cat qubit analysis tool.

Run with:
    python main.py

Menu options
------------
  1. Generate phase space diagram
  2. Show Liouvillian eigenmodes
  3. Show vacuum leakage analysis
  4. Change parameters
  0. Exit
"""

import os
import sys

# Ensure the project root is on the path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from core.system      import KerrCatSystem
from analysis.phase_space import PhaseSpaceDiagram
from analysis.eigenmodes  import LiouvillianModes
from analysis.leakage     import LeakageAnalysis
from analysis.mode_ratio  import ModeRatioAnalysis

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _banner():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        Delta-Kerr-Cat Qubit Analysis Tool        ║")
    print("╚══════════════════════════════════════════════════╝")


def _show_params(system: KerrCatSystem):
    print()
    print("  Current point parameters:")
    print(f"    P/K     = {system.P_over_K}")
    print(f"    Delta/K = {system.delta_over_K}")
    print(f"    N       = {system.N}")
    print(f"    n_th    = {system.n_th}")
    print(f"    alpha   = sqrt(P/K) = {system.cat_amplitude:.3f}")
    print()


def _menu():
    print("  ─────────────────────────────────────────────────")
    print("  1.  Generate phase space diagram")
    print("  2.  Show Liouvillian eigenmodes")
    print("  3.  Show vacuum leakage analysis")
    print("  4.  Change parameters")
    print("  5.  Mode ratio heatmap (3-solution region)")
    print("  0.  Exit")
    print("  ─────────────────────────────────────────────────")
    return input("  Select: ").strip()


def _prompt_float(label: str, current) -> float:
    raw = input(f"    {label} [{current}]: ").strip()
    return float(raw) if raw else current


def _prompt_int(label: str, current) -> int:
    raw = input(f"    {label} [{current}]: ").strip()
    return int(raw) if raw else current


def _change_params(system: KerrCatSystem) -> KerrCatSystem:
    """Prompt the user to update operating-point parameters."""
    print("\n  Enter new values (press Enter to keep current):")
    P     = _prompt_float("P/K",     system.P_over_K)
    delta = _prompt_float("Delta/K", system.delta_over_K)
    N     = _prompt_int  ("N",       system.N)
    n_th  = _prompt_float("n_th",    system.n_th)
    return KerrCatSystem(P, delta, N, n_th)


# ─────────────────────────────────────────────────────────────────────────────
# Action handlers
# ─────────────────────────────────────────────────────────────────────────────

def _run_phase_space():
    """Interactive phase-space generation with configurable sweep params."""
    print("\n  Phase space sweep parameters (press Enter for defaults):")
    N      = _prompt_int  ("N (Hilbert size)",      60)
    n_th   = _prompt_float("n_th",                  0.05)
    P_min  = _prompt_float("P/K min",               0.0)
    P_max  = _prompt_float("P/K max",               4.0)
    P_step = _prompt_float("P/K step",              0.02)
    d_min  = _prompt_float("Delta/K min",           -10.0)
    d_max  = _prompt_float("Delta/K max",            10.0)
    d_step = _prompt_float("Delta/K step",           0.1)

    diag = PhaseSpaceDiagram(
        N=N, n_th=n_th,
        P_range=(P_min, P_max), delta_range=(d_min, d_max),
        P_step=P_step, delta_step=d_step,
    )
    diag.generate()

    _ensure_output()
    save = os.path.join(OUTPUT_DIR, f"phase_diagram_N{N}_nth{n_th:.2f}.png")
    diag.plot(save_path=save)


def _run_eigenmodes(system: KerrCatSystem):
    """Compute and plot Liouvillian eigenmodes for current parameters."""
    n_modes = _prompt_int("Number of modes to compute", 6)

    modes = LiouvillianModes(system)
    modes.compute(n_modes=n_modes)
    modes.print_eigenvalues()

    _ensure_output()
    save = os.path.join(
        OUTPUT_DIR,
        f"eigenmodes_P{system.P_over_K}_D{system.delta_over_K}_N{system.N}.png",
    )
    modes.plot_wigner(save_path=save)


def _run_mode_ratio(system: KerrCatSystem):
    """Run mode-ratio heatmap sweep using current N and n_th."""
    print(f"\n  Mode ratio sweep  (N={system.N}, n_th={system.n_th} — from current params)")
    print("  Sweep grid parameters (press Enter for defaults):")
    P_min  = _prompt_float("P/K min",       0.5)
    P_max  = _prompt_float("P/K max",       3.0)
    P_step = _prompt_float("P/K step",      0.1)
    d_min  = _prompt_float("Delta/K min",   0.0)
    d_max  = _prompt_float("Delta/K max",  15.0)
    d_step = _prompt_float("Delta/K step",  0.3)

    mr = ModeRatioAnalysis(
        N=system.N, n_th=system.n_th,
        P_range=(P_min, P_max), delta_range=(d_min, d_max),
        P_step=P_step, delta_step=d_step,
    )
    mr.run()

    _ensure_output()
    save = os.path.join(
        OUTPUT_DIR,
        f"mode_ratio_N{system.N}_nth{system.n_th:.2f}.png",
    )
    mr.plot(save_path=save)


def _run_leakage(system: KerrCatSystem):
    """Run vacuum leakage analysis for current parameters."""
    T_max_factor = _prompt_int  ("T_max factor (T_max = factor × tau2)", 8)
    n_steps      = _prompt_int  ("Number of time steps",                500)
    late_frac    = _prompt_float("Late-window fit start fraction",       0.3)

    leak = LeakageAnalysis(system)
    leak.run(T_max_factor=T_max_factor, n_steps=n_steps)
    leak.fit(late_start_frac=late_frac)
    leak.print_summary()

    _ensure_output()
    save = os.path.join(
        OUTPUT_DIR,
        f"leakage_P{system.P_over_K}_D{system.delta_over_K}_N{system.N}.png",
    )
    leak.plot(save_path=save)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Default operating point
    system = KerrCatSystem(P_over_K=1.35, delta_over_K=6.0, N=40, n_th=0.05)

    _banner()

    while True:
        _show_params(system)
        choice = _menu()

        if choice == "1":
            _run_phase_space()

        elif choice == "2":
            _run_eigenmodes(system)

        elif choice == "3":
            _run_leakage(system)

        elif choice == "4":
            system = _change_params(system)
            print(f"\n  Parameters updated: {system}")

        elif choice == "5":
            _run_mode_ratio(system)

        elif choice == "0":
            print("\n  Goodbye.\n")
            break

        else:
            print("  Invalid choice — please enter 0, 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    # Windows multiprocessing safety guard
    from multiprocessing import freeze_support
    freeze_support()
    main()
