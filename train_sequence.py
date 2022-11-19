from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma
from sklearn.decomposition import PCA

from rate_network import simulate, tanh, generate_gaussian_pulse

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, help='Prefactor for L1 penalty on loss function')
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')

args = parser.parse_args()
print(args)

POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP_ITERS = 250 # Number of times to simulate network and plasticity rules per loss function evaluation
STD_EXPL = args.std_expl
L1_PENALTY = args.l1_pen

T = 0.1 # Total duration of one network simulation
dt = 1e-4 # Timestep
t = np.linspace(0, T, int(T / dt))
n_e = 15 # Number excitatory cells in sequence (also length of sequence)
n_i = 20 # Number inhibitory cells

# Make directory for outputting simulations
if not os.path.exists('sims_out'):
	os.mkdir('sims_out')

# Make subdirectory for this particular experiment
time_stamp = str(datetime.now()).replace(' ', '_')
out_dir = f'sims_out/seq_STD_EXPL_{STD_EXPL}_L1_PENALTY_{L1_PENALTY}_{time_stamp}'
os.mkdir(out_dir)
os.mkdir(os.path.join(out_dir, 'outcmaes'))

layer_colors = get_ordered_colors('winter', 15)

rule_names = [ # Define labels for all rules to be run during simulations
	r'',
	r'$y$',
	# r'$x$',
	r'$y^2$',
	# r'$x^2$',
	r'$x \, y$',
	r'$x \, y^2$',
	# r'$x^2 \, y$',
	# r'$x^2 \, y^2$',
	r'$y_{int}$',
	# r'$x \, y_{int}$',
	r'$x_{int}$',
	r'$x_{int} \, y$',

	r'$w$',
	r'$w \, y$',
	# r'$w \, x$',
	r'$w \, y^2$',
	# r'$w \, x^2$',
	r'$w \, x \, y$',
	r'$w \, x \, y^2$',
	# r'$w \, x^2 \, y$',
	# r'$w \, x^2 \, y^2$',
	r'$w y_{int}$',
	# r'$w x \, y_{int}$',
	r'$w x_{int}$',
	r'$w x_{int} \, y$',

	# r'$w^2$',
	# r'$w^2 \, y$',
	# r'$w^2 \, x$',
	# r'$w^2 \, y^2$',
	# r'$w^2 \, x^2$',
	# r'$w^2 \, x \, y$',
	# r'$w^2 \, x \, y^2$',
	# r'$w^2 \, x^2 \, y$',
	# r'$w^2 \, x^2 \, y^2$',
	# r'$w^2 y_{int}$',
	# r'$w^2 x \, y_{int}$',
	# r'$w^2 x_{int}$',
	# r'$w^2 x_{int} \, y$',
]

rule_names = [
	[r'$E \rightarrow E$ ' + r_name for r_name in rule_names],
	[r'$E \rightarrow I$ ' + r_name for r_name in rule_names],
	[r'$I \rightarrow E$ ' + r_name for r_name in rule_names],
]
rule_names = np.array(rule_names).flatten()

# Define input for activation of the network
r_in = np.zeros((len(t), n_e + n_i))
r_in[:, 0] = generate_gaussian_pulse(t, 5e-3, 5e-3, w=0.012) # Drive first excitatory cell with Gaussian input

transfer_e = partial(tanh, v_th=0.1)
transfer_i = partial(tanh, v_th=0.1)

w_e_e = 0.8e-3 / dt
w_e_i = 0.5e-4 / dt
w_i_e = -0.25e-4 / dt

# Define r_target, the target dynamics for the network to produce. 

r_target = np.zeros((len(t), n_e))
delay = 3.75e-3
period = 10e-3
offset = 2e-3

for i in range(n_e):
	active_range = (delay * i + offset, delay * i + period + offset)
	n_t_steps = int(period / dt)
	t_step_start = int(active_range[0] / dt)
	r_target[t_step_start:(t_step_start + n_t_steps), i] = 0.25 * np.sin(np.pi/period * dt * np.arange(n_t_steps))

def make_network():
	'''
	Generates an excitatory chain with recurrent inhibition and weak recurrent excitation. Weights that form sequence are distored randomly.

	'''
	w_initial = np.zeros((n_e + n_i, n_e + n_i))
	w_initial[:n_e, :n_e] = w_e_e * np.diag( 0.8 * np.log10(np.arange(n_e - 1) + 10), k=-1)
	w_initial[:n_e, :n_e] = w_initial[:n_e, :n_e] * (0.3 + 0.7 * np.random.rand(n_e, n_e))

	w_initial[:n_e, :n_e] = np.where(
		np.diag(np.ones(n_e - 1), k=-1) > 0,
		w_initial[:n_e, :n_e],
		exp_if_under_val(0.5, (n_e, n_e), 0.03 * w_e_e)
	)

	w_initial[n_e:, :n_e] = gaussian_if_under_val(0.8, (n_i, n_e), w_e_i, 0.3 * w_e_i)
	w_initial[:n_e, n_e:] = gaussian_if_under_val(0.8, (n_e, n_i), w_i_e, 0.3 * np.abs(w_i_e))

	return w_initial

def l2_loss(r : np.ndarray, r_target : np.ndarray):
	'''
	Calculates SSE between r, network activity, and r_target, target network activity
	'''
	if np.isnan(r).any():
		return 100000
	return np.sum(np.square(r[:, :n_e] - r_target))

def plot_results(results, eval_tracker, out_dir, title, plasticity_coefs):
	scale = 3
	n_res_to_show = BATCH_SIZE

	gs = gridspec.GridSpec(2 * n_res_to_show + 2, 2)
	fig = plt.figure(figsize=(4  * scale, (2 * n_res_to_show + 2) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(2 * n_res_to_show)]
	axs += [fig.add_subplot(gs[2 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 1, :])]

	for i in np.arange(BATCH_SIZE):
		# for each network in the batch, graph its excitatory, inhibitory activity, as well as the target activity
		r, w, w_initial = results[i]

		for l_idx in range(r.shape[1]):
			if l_idx < n_e:
				if l_idx % 1 == 0:
					axs[2 * i][0].plot(t, r[:, l_idx], c=layer_colors[l_idx % len(layer_colors)]) # graph excitatory neuron activity
					axs[2 * i][0].plot(t, r_target[:, l_idx], '--', c=layer_colors[l_idx % len(layer_colors)]) # graph target activity
			else:
				axs[2 * i][1].plot(t, r[:, l_idx], c='black') # graph inh activity

		vmin = np.min([w_initial.min(), w.min()])
		vmax = np.max([w_initial.max(), w.max()])

		mappable = axs[2 * i + 1][0].matshow(w_initial, vmin=vmin, vmax=vmax) # plot initial weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][0])

		mappable = axs[2 * i + 1][1].matshow(w, vmin=vmin, vmax=vmax) # plot final weight matrix
		plt.colorbar(mappable, ax=axs[2 * i + 1][1])


		axs[2 * i][0].set_title(title)
		for i_axs in range(2):
			axs[2 * i][i_axs].set_xlabel('Time (s)')
			axs[2 * i][i_axs].set_ylabel('Firing rate')

	# plot the coefficients assigned to each plasticity rule
	plasticity_coefs_abs = np.abs(plasticity_coefs)
	plasticity_coefs_argsort = np.flip(np.argsort(plasticity_coefs_abs))
	axs[2 * n_res_to_show + 1].bar(np.arange(len(plasticity_coefs)), plasticity_coefs[plasticity_coefs_argsort])
	axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(plasticity_coefs)))
	axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[plasticity_coefs_argsort], rotation=60, ha='right')
	axs[2 * n_res_to_show + 1].set_xlim(-1, len(plasticity_coefs))

	# plot the 10 larget coefficients applied to plasticity rules
	to_show = 10
	plasticity_coefs_argsort = plasticity_coefs_argsort[:to_show]
	axs[2 * n_res_to_show].bar(np.arange(to_show), plasticity_coefs[plasticity_coefs_argsort])
	axs[2 * n_res_to_show].set_xticks(np.arange(to_show))
	axs[2 * n_res_to_show].set_xticklabels(rule_names[plasticity_coefs_argsort], rotation=60, ha='right')
	axs[2 * n_res_to_show].set_xlim(-1, to_show)


	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']

	fig.tight_layout()
	fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
	plt.close('all')

def simulate_single_network(index, plasticity_coefs):
	'''
	Simulate one set of plasticity rules. `index` describes the simulation's position in the current batch and is used to randomize the random seed.
	'''
	for k in range(index):
		np.random.rand()

	w_initial = make_network() # make a new, distorted sequence
	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	for i in range(N_INNER_LOOP_ITERS):
		# below, simulate one activation of the network for the period T
		r, s, v, w_out = simulate(t, n_e, n_i, r_in + 4e-6 / dt * np.random.rand(len(t), n_e + n_i), transfer_e, transfer_i, plasticity_coefs, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1)
		if np.isnan(r).any(): # if simulation turns up nans in firing rate matrix, end the simulation
			return r, w, w_initial
		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	return r, w, w_initial

# Function to minimize (including simulation)

def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None):
	start = time.time()

	for k in range(BATCH_SIZE):
		np.random.rand()

	pool = mp.Pool(POOL_SIZE)
	f = partial(simulate_single_network, plasticity_coefs=plasticity_coefs)
	results = pool.map(f, np.arange(BATCH_SIZE))
	pool.close()

	loss = np.sum([l2_loss(res[0], r_target) for res in results]) + L1_PENALTY * BATCH_SIZE * np.sum(np.abs(plasticity_coefs))

	if eval_tracker is not None:
		if np.isnan(eval_tracker['best_loss']) or loss < eval_tracker['best_loss']:
			if eval_tracker['evals'] > 0:
				eval_tracker['best_loss'] = loss
		plot_results(results, eval_tracker, out_dir, f'Loss: {loss}\n', plasticity_coefs)
		eval_tracker['evals'] += 1

	dur = time.time() - start
	print('duration:', dur)
	print('guess:', plasticity_coefs)
	print('loss:', loss)
	print('')

	return loss

x0 = np.zeros(48)


# def set_smallest_n_zero(arr, n):
# 	arr_copy = copy(arr)
# 	sort_indices = np.argsort(np.abs(arr))
# 	print(sort_indices)
# 	for i, sort_i in enumerate(sort_indices):
# 		if sort_i >= (len(arr) - n):
# 			arr_copy[i] = 0
# 	return arr_copy


eval_tracker = {
	'evals': 0,
	'best_loss': np.nan,
}

simulate_plasticity_rules(x0, eval_tracker=eval_tracker)

options = {
	'verb_filenameprefix': os.path.join(out_dir, 'outcmaes/'),
}

x, es = cma.fmin2(
	partial(simulate_plasticity_rules, eval_tracker=eval_tracker),
	x0,
	STD_EXPL,
	restarts=10,
	bipop=True,
	options=options)

print(x)
print(es.result_pretty())
