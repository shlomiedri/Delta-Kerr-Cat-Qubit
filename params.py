"""
System parameters from Table VII.

Conventions:
- Quantities named `*_over_2pi` are in Hz.
- Quantities without that suffix are angular rates in rad/s.
- Time quantities are in seconds.
"""

import numpy as np
#Simulation constant
N= 30


# Unit helpers
HZ = 1.0
KHZ = 1e3 * HZ
MHZ = 1e6 * HZ
GHZ = 1e9 * HZ
US = 1e-6


# Frequencies/rates reported as x / (2*pi)
omega_a_over_2pi = 6.0 * GHZ
omega_b_over_2pi = 8.9 * GHZ

kappa_b_c_over_2pi = 1.4 * MHZ
kappa_b_l_over_2pi = 0.5 * MHZ

# User-requested change: K = 1.2 MHz (instead of table value 6.7 MHz)
K_over_2pi = 1.2 * MHZ
g3_over_2pi = 20.0 * MHZ

chi_ab_min_over_2pi = 200.0 * KHZ
chi_ab_max_over_2pi = 250.0 * KHZ

# Cat-Rabi drive strength ε_x (not the squeezing drive)
eps_x_over_2pi = 6.5 * MHZ
eps_x_alt_over_2pi = 740.0 * KHZ

# Squeezing (two-photon) drive strength P
P_over_2pi = 17.75 * MHZ
P_alt_over_2pi = 15.5 * MHZ

delta_over_2pi = 2.2 * MHZ
g_cr_over_2pi = 1.7 * MHZ


# Convert to angular units (rad/s)
omega_a = 2 * np.pi * omega_a_over_2pi
omega_b = 2 * np.pi * omega_b_over_2pi

kappa_b_c = 2 * np.pi * kappa_b_c_over_2pi
kappa_b_l = 2 * np.pi * kappa_b_l_over_2pi
kappa_b = kappa_b_c + kappa_b_l

K = 2 * np.pi * K_over_2pi
g3 = 2 * np.pi * g3_over_2pi

chi_ab_min = 2 * np.pi * chi_ab_min_over_2pi
chi_ab_max = 2 * np.pi * chi_ab_max_over_2pi

eps_x = 2 * np.pi * eps_x_over_2pi
eps_x_alt = 2 * np.pi * eps_x_alt_over_2pi

P = 2 * np.pi * P_over_2pi
P_alt = 2 * np.pi * P_alt_over_2pi

delta= 2 * np.pi * delta_over_2pi
g_cr = 2 * np.pi * g_cr_over_2pi


# Coherence/decay times
T_2 = 3.4 * US
T_2e = 13.7 * US

# Table value: T1 = 15.5 us, but expressed via the requested relation T_1 = 1 / kappa_1
kappa_1 = 1.0 / (15.5 * US)
T_1 = 1.0 / kappa_1


# User-requested thermal occupancy
n_th = 0.05


SYSTEM_PARAMS = {
    "omega_a": omega_a,
    "omega_b": omega_b,
    "kappa_b_c": kappa_b_c,
    "kappa_b_l": kappa_b_l,
    "kappa_b": kappa_b,
    "K": K,
    "g3": g3,
    "chi_ab_min": chi_ab_min,
    "chi_ab_max": chi_ab_max,
    "eps_x": eps_x,
    "eps_x_alt": eps_x_alt,
    "P": P,
    "P_alt": P_alt,
    "delta_as": delta,
    "g_cr": g_cr,
    "kappa_1": kappa_1,
    "T_1": T_1,
    "T_2": T_2,
    "T_2e": T_2e,
    "n_th": n_th,
}
