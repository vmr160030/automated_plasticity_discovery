from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val, rev_argsort
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
N_INNER_LOOP_RANGE = (150, 400) # Number of times to simulate network and plasticity rules per loss function evaluation
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
out_dir = f'sims_out/seq_ei_STD_EXPL_{STD_EXPL}_L1_PENALTY_{L1_PENALTY}_{time_stamp}'
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

	np.fill_diagonal(w_initial, 0)

	return w_initial

def l2_loss(r : np.ndarray, r_target : np.ndarray):
	'''
	Calculates SSE between r, network activity, and r_target, target network activity
	'''
	if np.isnan(r).any():
		return 1e8
	return np.sum(np.square(r[:, :n_e] - r_target))

def plot_results(results, eval_tracker, out_dir, title, plasticity_coefs):
	scale = 3
	n_res_to_show = BATCH_SIZE

	gs = gridspec.GridSpec(2 * n_res_to_show + 2, 2)
	fig = plt.figure(figsize=(4  * scale, (2 * n_res_to_show + 2) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(2 * n_res_to_show)]
	axs += [fig.add_subplot(gs[2 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 1, :])]

	all_effects = []

	for i in np.arange(BATCH_SIZE):
		# for each network in the batch, graph its excitatory, inhibitory activity, as well as the target activity
		r, w, w_initial, cumulative_loss, effects = results[i]

		all_effects.append(effects)

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

	### plot the coefficients assigned to each plasticity rule
	# plasticity_coefs_abs = np.abs(plasticity_coefs)
	# plasticity_coefs_argsort = np.flip(np.argsort(plasticity_coefs_abs))
	# axs[2 * n_res_to_show + 1].bar(np.arange(len(plasticity_coefs)), plasticity_coefs[plasticity_coefs_argsort])
	# axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(plasticity_coefs)))
	# axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[plasticity_coefs_argsort], rotation=60, ha='right')
	# axs[2 * n_res_to_show + 1].set_xlim(-1, len(plasticity_coefs))
	partial_rules_len = int(len(plasticity_coefs) / 3)

	print(np.array(all_effects).shape)
	effects = np.mean(np.array(all_effects), axis=0)

	axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(effects)))
	effects_argsort = []
	for l in range(3):
		effects_partial = effects[l * partial_rules_len: (l+1) * partial_rules_len]
		effects_argsort_partial = np.flip(np.argsort(effects_partial))
		effects_argsort.append(effects_argsort_partial + l * partial_rules_len)
		axs[2 * n_res_to_show + 1].bar(np.arange(len(effects_argsort_partial)) + l * 16, effects_partial[effects_argsort_partial] / np.max(np.abs(effects_partial)))
	axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[np.concatenate(effects_argsort)], rotation=60, ha='right')
	axs[2 * n_res_to_show + 1].set_xlim(-1, len(effects))

	# plot the coefficients assigned to each plasticity rule (unsorted by size)
	for l in range(3):
		axs[2 * n_res_to_show].bar(np.arange(partial_rules_len) + l * partial_rules_len, plasticity_coefs[l * partial_rules_len: (l+1) * partial_rules_len])
	axs[2 * n_res_to_show].set_xticks(np.arange(len(plasticity_coefs)))
	axs[2 * n_res_to_show].set_xticklabels(rule_names, rotation=60, ha='right')
	axs[2 * n_res_to_show].set_xlim(-1, len(plasticity_coefs))

	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']

	fig.tight_layout()
	fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
	plt.close('all')

def simulate_single_network(index, plasticity_coefs, gamma=0.98, track_params=False):
	'''
	Simulate one set of plasticity rules. `index` describes the simulation's position in the current batch and is used to randomize the random seed.
	'''
	np.random.seed()

	w_initial = make_network() # make a new, distorted sequence
	n_inner_loop_iters = np.random.randint(N_INNER_LOOP_RANGE[0], N_INNER_LOOP_RANGE[1])

	w = copy(w_initial)
	w_plastic = np.where(w != 0, 1, 0).astype(int) # define non-zero weights as mutable under the plasticity rules

	cumulative_loss = 0

	all_effects = np.zeros(plasticity_coefs.shape)

	for i in range(n_inner_loop_iters):
		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects = simulate(t, n_e, n_i, r_in + 4e-6 / dt * np.random.rand(len(t), n_e + n_i), transfer_e, transfer_i, plasticity_coefs, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1, track_params=track_params)

		loss = l2_loss(r, r_target)
		cumulative_loss = cumulative_loss * gamma + loss

		if np.isnan(r).any(): # if simulation turns up nans in firing rate matrix, end the simulation
			break

		if effects is not None:
			all_effects += effects

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation

	return r, w, w_initial, cumulative_loss / (1 / np.log(1/gamma)), all_effects

# Function to minimize (including simulation)

def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None):
	start = time.time()

	pool = mp.Pool(POOL_SIZE)
	f = partial(simulate_single_network, plasticity_coefs=plasticity_coefs, track_params=False)
	results = pool.map(f, np.arange(BATCH_SIZE))
	pool.close()

	loss = np.sum([res[3] for res in results]) + L1_PENALTY * BATCH_SIZE * np.sum(np.abs(plasticity_coefs))

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

eval_tracker = {
	'evals': 0,
	'best_loss': np.nan,
}


# x1 = str('-0.015207665210977462 0.003078088587819452 -0.0039007094294291576 0.008599621317100665 -0.00047642194349143633 0.00602048918919687 -0.0018716519801185597 -0.00016293863856126447 0.0057603159304138 0.007164097721079621 0.0019926294159914814 0.00595263793866414 -0.017403466587404763 -0.00883200546190559 0.009635066819565215 -0.004151492460011847 0.019988817255316767 -0.012569949137790403 -0.0037411279464771383 0.010966351408568718 0.0009625506133517105 0.0039495421457438114 0.006042695238710668 -8.5911785675411e-05 -0.005326489669087308 0.022683356658544138 -0.010409137146929471 -0.0011139838503831192 0.005356298230503296 -0.007884795222558989 -0.022861228121030214 -0.0015895935921566495 0.011752543411873664 0.00600517448624817 -0.00970352095865758 -0.010487083106439879 -0.021987895722459377 -0.005914623268820082 -0.0016538721120160343 0.010385351694610353 -0.004344927872738611 -0.015932842330299665 -0.0013563186199331855 0.010122022572432051 0.0011592418848316837 0.01571435193066515 -0.012612649067515008 -0.0003204793986235654').split(' ')
# x1 = np.array([float(n) for n in x1])

# effect_sizes = np.array([1.52863724e+05, 3.10151072e+04, 1.19831736e+05, 3.50751820e+02,
#  1.02272069e+04, 7.20597267e+04, 7.21772729e+04, 2.45813518e+04,
#  4.11830991e+04, 1.97110102e+04, 3.30041244e+04, 2.32333605e+04,
#  6.68113294e+02, 2.02498156e+05, 3.01625134e+05, 1.64388333e+04,
#  2.50752412e+05, 2.41782677e+05, 2.76933231e+05, 1.44527098e+05,
#  2.63289899e+05, 2.26216289e+06, 2.02114928e+05, 1.78664154e+05,
#  1.27894762e+05, 1.50168631e+04, 3.09739693e+05, 1.12002602e+05,
#  2.08145925e+04, 3.32664737e+06, 2.96744597e+05, 1.35967389e+05,
#  9.95835279e+05, 2.78712570e+04, 2.54634958e+05, 5.35908805e+04,
#  1.50543614e+05, 1.65921989e+05, 1.64496460e+05, 2.17419614e+04,
#  7.47261217e+04, 7.70435349e+03, 9.80963610e+03, 5.97335150e+03,
#  1.16657838e+04, 7.43509599e+04, 1.09951165e+04, 1.80412273e+04,])

# num_to_silence = [13, 14, 14]
# for j in range(3):
# 	st = 16 * j
# 	en = 16 * (j + 1)
# 	effects_partial = effect_sizes[st:en]
# 	set_smallest_n_zero(effects_partial, num_to_silence[j], arr_set=x1[st:en])

# print(x1)

# simulate_plasticity_rules(x1, eval_tracker=eval_tracker)
# simulate_plasticity_rules(x1, eval_tracker=eval_tracker)
# simulate_plasticity_rules(x1, eval_tracker=eval_tracker)

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
