import qutip as qt 
import numpy as np 
import os 
import params as g



def make_operators(N=g.N):
    a = qt.destroy(N)
    a_dag = a.dag()
    return {'a': a, 'a_dag': a_dag, 'n': a_dag * a,
            'x': (a + a_dag) / np.sqrt(2),
            'p': 1j * (a_dag - a) / np.sqrt(2)}


def make_hamiltonian(ops, K=g.K, P=g.P, delta=g.delta):
    """Normalised by g.K so matrix elements are O(1)."""
    kerr   = -(K/g.K)     * (ops['a_dag']**2 * ops['a']**2)
    pump   =  (P/g.K)     * (ops['a']**2 + ops['a_dag']**2)
    detune =  (delta/g.K) * ops['a_dag'] * ops['a']
    return kerr + pump + detune


def make_cat_states(N=g.N, P=g.P, K=g.K):
    alpha     = np.sqrt(P / K)
    coh_plus  = qt.coherent(N,  alpha)
    coh_minus = qt.coherent(N, -alpha)
    return {'alpha':     alpha,
            'coh_plus':  coh_plus,
            'coh_minus': coh_minus,
            'cat_plus':  (coh_plus + coh_minus).unit(),
            'cat_minus': (coh_plus - coh_minus).unit()}


def make_collapse_ops(ops, kappa1=g.kappa_1, nth=g.n_th):
    """Collapse ops normalised by g.K to match dimensionless Hamiltonian."""
    return [np.sqrt((kappa1 / g.K )* (1 + nth)) * ops['a'],
            np.sqrt((kappa1 / g.K ) * nth)        * ops['a_dag']] # Single photon lose, Thermal bath 


def make_logical_paulis(cats):
    cp = cats['cat_plus']  * cats['cat_plus'].dag()
    cm = cats['cat_minus'] * cats['cat_minus'].dag()
    ap = cats['coh_plus']  * cats['coh_plus'].dag()
    am = cats['coh_minus'] * cats['coh_minus'].dag()
    return {'Z_L': cp - cm, 'X_L': ap - am, 'Id_L': cp + cm}
