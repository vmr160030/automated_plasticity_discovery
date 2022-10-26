from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
import matplotlib.pyplot as plt
from datetime import datetime

import cma
from sklearn.decomposition import PCA

from rate_network import simulate_seq, tanh, generate_gaussian_pulse

if not os.path.exists('./sims_out'):
	os.mkdir('./sims_out')

out_dir = f'./sims_out/{datetime.now()}'
os.mkdir(out_dir)

layer_colors = get_ordered_colors('winter', 20)
run_colors = get_ordered_colors('hot', 20)

T = 0.1
dt = 1e-4
t = np.linspace(0, T, int(T / dt))
n_e = 15
n_i = 10

r_in = generate_gaussian_pulse(t, 0.005, 0.005, w=1)

transfer_e = partial(tanh, v_th=0.1)
transfer_i = partial(tanh, v_th=0.5)

plasticity_coefs = np.zeros(81)

w_e_e = 4e-4 / dt
w_e_i = 1e-4 / dt
w_i_e = -2.5e-5 / dt

w_initial = np.zeros((n_e + n_i, n_e + n_i))
w_initial[:n_e, :n_e] = w_e_e * np.diag(np.ones(n_e - 1), k=-1) + 0.05 * np.ones((n_e, n_e))
w_initial[n_e:, :n_e] = w_e_i * np.ones((n_i, n_e))
w_initial[:n_e, n_e:] = w_i_e * np.ones((n_e, n_i))

# Defining L2 loss and objective function

r_target = np.zeros((len(t), n_e))
period = 6e-3

for i in range(n_e):
	active_range = (period * i, period * (i+1))
	n_t_steps = int(period / dt)
	t_step_start = int(active_range[0] / dt)
	r_target[t_step_start:(t_step_start + n_t_steps), i] = np.sin(np.pi/period * dt * np.arange(n_t_steps))

def l2_loss(r, r_target):
	return np.sum(np.square(r[:, :n_e] - r_target))

eval_tracker = {
	'evals': 0,
	'best_loss': np.nan,
}

# Function to minimize (including simulation)

def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None):
	start = time.time()

	w = copy(w_initial)
	for i in range(50):
		r, s, v, w_out = simulate_seq(t, n_e, n_i, r_in, transfer_e, transfer_i, plasticity_coefs, w, dt=dt, tau_e=5e-3, tau_i=0.1e-3, g=1, w_u=1)
		w = w_out

	loss = l2_loss(r, r_target) + 100 * np.sum(np.abs(plasticity_coefs))

	if eval_tracker is not None:
		scale = 1
		fig, axs = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10 * scale, 3  * scale))

		for l_idx in range(r.shape[1]):
			if l_idx < n_e:
				if l_idx % 1 == 0:
					axs[0, 0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)])
					axs[0, 0].plot(t, r_target[:, l_idx], '--', c=layer_colors[l_idx % len(layer_colors)])
			else:
				axs[0, 1].plot(t, r[:, l_idx], c='black')

		axs[1, 0].matshow(w_initial)
		axs[1, 1].matshow(w)

		axs[0, 0].set_title(f'Loss: {loss}')

		pad = 4 - len(str(eval_tracker['evals']))
		zero_padding = '0' * pad
		evals = eval_tracker['evals']

		fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')

		eval_tracker['best_loss'] = loss
		eval_tracker['evals'] += 1

	dur = time.time() - start
	print('duration:', dur)
	print('guess:', plasticity_coefs)
	print(r[:, :n_e])
	print('loss:', loss)
	print('')

	return loss

simulate_plasticity_rules(np.zeros(81), eval_tracker=eval_tracker)

x0 = np.zeros(81)
x, es = cma.fmin2(partial(simulate_plasticity_rules, eval_tracker=eval_tracker), x0, 0.1)
print(x)
print(es.result_pretty())


