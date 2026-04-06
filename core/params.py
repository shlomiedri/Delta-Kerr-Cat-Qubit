"""
core/params.py
--------------
Physical constants for the Delta-Kerr-Cat qubit system.
All values are sourced from Table VII of the experimental paper.

Conventions:
  - Angular rates in rad/s (multiply by 2*pi from Hz values)
  - Time quantities in seconds
  - Subscript _over_2pi means the value is f = omega/(2*pi) in Hz
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Unit helpers
# ─────────────────────────────────────────────────────────────────────────────
HZ  = 1.0
KHZ = 1e3
MHZ = 1e6
GHZ = 1e9
US  = 1e-6          # microsecond in seconds

# ─────────────────────────────────────────────────────────────────────────────
# Cavity frequencies
# ─────────────────────────────────────────────────────────────────────────────
omega_a = 2 * np.pi * 6.0 * GHZ   # cavity a frequency  (rad/s)
omega_b = 2 * np.pi * 8.9 * GHZ   # cavity b frequency  (rad/s)

# ─────────────────────────────────────────────────────────────────────────────
# Dissipation rates
# ─────────────────────────────────────────────────────────────────────────────
kappa_b_c = 2 * np.pi * 1.4 * MHZ   # cavity b coupling loss  (rad/s)
kappa_b_l = 2 * np.pi * 0.5 * MHZ   # cavity b internal loss  (rad/s)
kappa_b   = kappa_b_c + kappa_b_l   # total cavity b loss      (rad/s)

# Single-photon decay of cavity a (T1 = 5 us -> kappa_1 = 0.2 MHz)
kappa_1 = 0.2 * MHZ                 # rad/s  (NOTE: not multiplied by 2pi,
                                     # consistent with the original codebase)

# ─────────────────────────────────────────────────────────────────────────────
# Nonlinear parameters
# ─────────────────────────────────────────────────────────────────────────────
# Kerr nonlinearity (user-requested: 1.2 MHz instead of table value 6.7 MHz)
K  = 2 * np.pi * 1.2 * MHZ          # rad/s

g3 = 2 * np.pi * 20.0 * MHZ         # third-order coupling  (rad/s)

chi_ab_min = 2 * np.pi * 200.0 * KHZ
chi_ab_max = 2 * np.pi * 250.0 * KHZ

g_cr = 2 * np.pi * 1.7 * MHZ        # cross-resonance coupling

# ─────────────────────────────────────────────────────────────────────────────
# Drive parameters
# ─────────────────────────────────────────────────────────────────────────────
P     = 2 * np.pi * 17.75 * MHZ     # two-photon squeezing drive (primary)
P_alt = 2 * np.pi * 15.5  * MHZ     # two-photon squeezing drive (alternate)

eps_x     = 2 * np.pi * 6.5   * MHZ   # cat-Rabi drive (primary)
eps_x_alt = 2 * np.pi * 0.740 * MHZ   # cat-Rabi drive (alternate)

delta = 2 * np.pi * 2.2 * MHZ       # detuning (rad/s)

# ─────────────────────────────────────────────────────────────────────────────
# Coherence times
# ─────────────────────────────────────────────────────────────────────────────
T_1  = 1.0 / kappa_1                # ~5 us
T_2  = 3.4  * US
T_2e = 13.7 * US

# ─────────────────────────────────────────────────────────────────────────────
# Default simulation settings
# ─────────────────────────────────────────────────────────────────────────────
N    = 30     # default Hilbert space truncation
n_th = 0.05   # default thermal photon occupancy
