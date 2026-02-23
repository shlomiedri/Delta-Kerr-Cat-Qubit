import params as g 
import qutip as qt 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def make_operators(N=g.N):
    """
    Parameters:
        N (int): Hilbert space dimension (number of Fock states).
    Returns:
        'a'     — annihilation operator â
        'a_dag' — creation operator â†
        'n'     — number operator n̂ = â†â
        'x'     — position quadrature x̂ = (â + â†) / √2
        'p'     — momentum quadrature p̂ = i(â† − â) / √2
    """ 
    a = qt.destroy(N)
    a_dag = a.dag()
    n = a_dag * a
    x = (a + a_dag) / np.sqrt(2)
    p = 1j * (a_dag - a) / np.sqrt(2)
    return {'a': a, 'a_dag': a_dag, 'n': n, 'x': x, 'p': p} 


def make_hamiltonian(ops,K = g.K, P = g.P, delta = g.delta):
    """
   Parameters:
        ops  (dict): output of make_operators()
        K    (float): Kerr nonlinearity strength
        eps2 (float): squeezing drive amplitude
        delta(float): detuning Δ
    Returns:
         the Hamiltonian operator
    """
    kerr = K*(ops['a']**2*ops['a_dag']**2)
    pump = P*(ops['a']**2 + ops['a_dag']**2)
    detune = delta*ops['a']*ops['a_dag']
    return kerr+pump+detune

def make_cat_states(N= g.N, P = g.P, K = g.K):
    """
    Parameters:
        N    (int): Hilbert space dimension
        eps2 (float): squeezing drive amplitude
        K    (float): Kerr nonlinearity
    Returns:
        'alpha'     — coherent state amplitude α = √(ε₂/K)
        'coh_plus'  — coherent state |+α⟩
        'coh_minus' — coherent state |−α⟩
        'cat_plus'  — even cat state |C⁺⟩  (logical |0̄⟩)
        'cat_minus' — odd cat state  |C⁻⟩  (logical |1̄⟩)
    """
    if P/K < 0 : 
        return "Ratio must be postive"
    alpha = np.sqrt(P/K)
    coh_plus = qt.coherent(N,alpha)
    coh_minus = qt.coherent(N,-alpha)
    cat_plus = (coh_plus+coh_minus).unit()
    cat_minus = (coh_plus-coh_minus).unit()
    return {"alpha": alpha, 
            "coh_plus": coh_plus,
            "coh_minus":coh_minus,
            "cat_plus": cat_plus,
            "cat_minus": cat_minus} 

def make_collapse_ops(ops, kappa1=g.kappa_1, nth=g.n_th, kappa_phi=1/g.T_2e):
    """
    Parameters:
        ops      (dict): output of make_operators()
        kappa1   (float): single-photon loss rate
        nth      (float): thermal photon number
        kappa_phi(float): pure dephasing rate

    Returns:
        list of qutip.Qobj: [c_loss, c_thermal, c_dephase]
    """
    c_loss = np.sqrt(kappa1+nth)*ops['a']
    c_thermal = np.sqrt(nth*kappa1)*ops['a_dag']
    c_dephase = np.sqrt(kappa_phi)*ops['n']
    return [c_loss, c_thermal, c_dephase ]

def make_logical_paulis(cats):
    cat_p = cats['cat_plus'] * cats['cat_plus'].dag() 
    cat_m = cats['cat_minus'] * cats['cat_minus'].dag()
    alpha_p = cats['coh_plus'] * cats['coh_plus'].dag() 
    alpha_m = cats['coh_minus'] * cats['coh_minus'].dag()
    Z_L = cat_p - cat_m 
    X_L = alpha_p - alpha_m
    Id_L = cat_p + cat_m 
    return {"Z_L": Z_L,
            "X_L": X_L,
            "Id_L":Id_L} 


def plot_T_X_vs_P():
    P_vals = np.linspace(0.5*g.P, 2*g.P, 20)
    T_X_vals = []

    for P in P_vals:
        ops   = make_operators()
        H     = make_hamiltonian(ops=ops, P=P, delta=0)
        c_ops = make_collapse_ops(ops)

        L = qt.liouvillian(H, c_ops)

        eigvals = L.eigenenergies(sparse=True, sort='low', eigvals=2)

        non_zero = eigvals[np.abs(eigvals) > 1e-10]
        mixing_rate = np.min(np.abs(non_zero.real))
        T_X_vals.append(1.0 / mixing_rate)

    plt.figure()
    plt.plot(P_vals , np.array(T_X_vals) * g.kappa_1)
    plt.xlabel(r'Squeezing drive strength $P$')
    plt.ylabel(r'$T_X$ (ms)')
    plt.title(r'$T_X$ from Liouvillian gap vs squeezing drive strength')
    plt.tight_layout()
    plt.show()
    return P_vals, T_X_vals


def plot_T_X_vs_delta():
    delta_vals = np.linspace(0.5*g.delta, 2*g.delta, 20)
    T_X_vals = []

    for delta in delta_vals:
        ops   = make_operators()
        H     = make_hamiltonian(ops=ops, delta=delta)
        c_ops = make_collapse_ops(ops)

        L = qt.liouvillian(H, c_ops)

        eigvals = L.eigenenergies(sparse=True, sort='low', eigvals=2)

        non_zero = eigvals[np.abs(eigvals) > 1e-10]
        mixing_rate = np.min(np.abs(non_zero.real))
        T_X_vals.append(1.0 / mixing_rate)

    plt.figure()
    plt.plot(delta_vals , np.array(T_X_vals) * 1e3)
    plt.xlabel(r'Detuning $\Delta$')
    plt.ylabel(r'$T_X$ (ms)')
    plt.title(r'$T_X$ from Liouvillian gap vs detuning')
    plt.tight_layout()
    plt.show()
    return delta_vals, T_X_vals



if __name__ == "__main__":
    p_vals = plot_T_X_vs_P()
    delta_vals = plot_T_X_vs_delta()