import numpy as np
from copy import deepcopy as copy

### For initiating activity

def generate_gaussian_pulse(t, u, s, w=1):
    return w / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * np.square((t-u) / s))

### Related to activation functions

def shift(x : np.ndarray):
    shifted = np.concatenate([[0], copy(x[:-1])])
    return shifted

def threshold_linear(s : np.ndarray, v_th : float):
    shifted_s = s - v_th
    shifted_s[shifted_s < 0] = 0
    return shifted_s

def tanh(s : np.ndarray, v_th : float):
    return np.tanh(threshold_linear(s, v_th))

def sigmoid(s : np.ndarray, v_th : float, spread : float):
    return 2 / (1 + np.exp(-1*(s - v_th) / spread))

def threshold_power(s : np.ndarray, v_th : float, p : float):
    return np.power(threshold_linear(s, v_th), p)

### Simulate dynamics

def simulate(t : np.ndarray, n_e : int, n_i : int, inp : np.ndarray, transfer_e, transfer_i, plasticity_coefs : np.ndarray, w : np.ndarray, tau_e=5e-3, tau_i=5e-3, tau_a=10e-3, dt=1e-6, g=1, w_u=1):    
    inh_activity = np.zeros((len(t)))
    r = np.zeros((len(t), n_e + n_i))
    s = np.zeros((len(t), n_e + n_i))
    v = np.zeros((len(t), n_e + n_i))

    w_copy = copy(w)

    tau = np.concatenate([tau_e * np.ones(n_e), tau_i * np.ones(n_i)])

    for i in range(0, len(t) - 1):
        v[i+1, :] = w_u * inp[i, :] + np.dot(w_copy, r[i, :].T)
        s[i+1, :] = s[i, :] + (v[i+1, :] - s[i, :]) * dt / tau

        r[i+1, :n_e] = g * transfer_e(s[i, :n_e])
        r[i+1, n_e:] = g * transfer_i(s[i, n_e:])

        # find cross products

        r_0_pow = np.ones(n_e + n_i)
        r_1_pow = r[i+1, :]
        r_2_pow = np.square(r[i+1, :])

        r_0_r_0 = np.outer(r_0_pow, r_0_pow)
        r_0_r_1 = np.outer(r_0_pow, r_1_pow)
        r_1_r_0 = r_0_r_1.T
        r_0_r_2 = np.outer(r_0_pow, r_2_pow)
        r_2_r_0 = r_0_r_2.T
        r_1_r_1 = np.outer(r_1_pow, r_1_pow)
        r_1_r_2 = np.outer(r_1_pow, r_2_pow)
        r_2_r_1 = r_1_r_2.T
        r_2_r_2 = np.outer(r_2_pow, r_2_pow)


        r_cross_products = np.stack([
        	r_0_r_0,
        	r_0_r_1,
        	r_1_r_0,
        	r_0_r_2,
        	r_2_r_0,
        	r_1_r_1,
        	r_1_r_2,
        	r_2_r_1,
        	r_2_r_2,
        ])

       	w_updates_unweighted = np.concatenate([r_cross_products, w_copy * r_cross_products, np.square(w_copy) * r_cross_products])
       	
        dw_e_e = np.sum(plasticity_coefs[:27].reshape(27, 1, 1) * w_updates_unweighted[:, :n_e, :n_e], axis=0)
        dw_e_i = np.sum(plasticity_coefs[27:54].reshape(27, 1, 1) * w_updates_unweighted[:, :n_e, n_e:], axis=0)
        dw_i_e = np.sum(plasticity_coefs[54:].reshape(27, 1, 1) * w_updates_unweighted[:, n_e:, :n_e], axis=0)

       	w_copy[:n_e, :n_e] += 0.0005 * dw_e_e
        w_copy[:n_e, n_e:] += 0.0005 * dw_e_i
        w_copy[n_e:, :n_e] += 0.0005 * dw_i_e

    return r, s, v, w_copy

