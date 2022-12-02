from copy import deepcopy as copy
import numpy as np
import os
import time
from functools import partial
from disp import get_ordered_colors
from aux import gaussian_if_under_val, exp_if_under_val, rev_argsort, set_smallest_n_zero
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import multiprocessing as mp
import argparse
import cma

from rate_network import simulate, tanh, generate_gaussian_pulse

### Parse arguments 

parser = argparse.ArgumentParser()
parser.add_argument('--std_expl', metavar='std', type=float, help='Initial standard deviation for parameter search via CMA-ES')
parser.add_argument('--l1_pen', metavar='l1', type=float, help='Prefactor for L1 penalty on loss function')
parser.add_argument('--dw_pen', metavar='dwp', type=float, help='Penalty for fraction of plasticity-induced weight change that occurs late in simulation')
parser.add_argument('--pool_size', metavar='ps', type=int, help='Number of processes to start for each loss function evaluation')
parser.add_argument('--batch', metavar='b', type=int, help='Number of simulations that should be batched per loss function evaluation')

args = parser.parse_args()
print(args)

POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch
N_INNER_LOOP_RANGE = (190, 200) # Number of times to simulate network and plasticity rules per loss function evaluation
STD_EXPL = args.std_expl
L1_PENALTY = args.l1_pen
DW_PENALTY = args.dw_pen
DW_LAG = 5

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
out_dir = f'sims_out/seq_ei_jit_accel_STD_EXPL_{STD_EXPL}_L1_PENALTY_{L1_PENALTY}_DW_PENALTY_{DW_PENALTY}_{time_stamp}'
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

w_e_e = 0.8e-3 / dt
w_e_i = 0.5e-4 / dt
w_i_e = -0.3e-4 / dt

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
	w_initial[:n_e, :n_e] = w_e_e * np.diag(0.8 * np.log10(np.arange(n_e - 1) + 10), k=-1)
	w_initial[:n_e, :n_e] = w_initial[:n_e, :n_e] * (0.2 + 1.5  * np.random.rand(n_e, n_e))

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

	gs = gridspec.GridSpec(2 * n_res_to_show + 3, 2)
	fig = plt.figure(figsize=(4  * scale, (2 * n_res_to_show + 3) * scale), tight_layout=True)
	axs = [[fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])] for i in range(2 * n_res_to_show)]
	axs += [fig.add_subplot(gs[2 * n_res_to_show, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 1, :])]
	axs += [fig.add_subplot(gs[2 * n_res_to_show + 2, :])]

	all_effects = []

	for i in np.arange(BATCH_SIZE):
		# for each network in the batch, graph its excitatory, inhibitory activity, as well as the target activity
		r, w, w_initial, normed_loss, effects, all_weight_deltas = results[i]

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

		axs[2 * n_res_to_show + 2].plot(np.arange(len(all_weight_deltas)), np.log(all_weight_deltas), label=f'{i}')

	### plot the coefficients assigned to each plasticity rule
	# plasticity_coefs_abs = np.abs(plasticity_coefs)
	# plasticity_coefs_argsort = np.flip(np.argsort(plasticity_coefs_abs))
	# axs[2 * n_res_to_show + 1].bar(np.arange(len(plasticity_coefs)), plasticity_coefs[plasticity_coefs_argsort])
	# axs[2 * n_res_to_show + 1].set_xticks(np.arange(len(plasticity_coefs)))
	# axs[2 * n_res_to_show + 1].set_xticklabels(rule_names[plasticity_coefs_argsort], rotation=60, ha='right')
	# axs[2 * n_res_to_show + 1].set_xlim(-1, len(plasticity_coefs))
	partial_rules_len = int(len(plasticity_coefs) / 3)

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

	axs[2 * n_res_to_show + 2].set_xlabel('Epochs')
	axs[2 * n_res_to_show + 2].set_ylabel('log(delta W)')
	axs[2 * n_res_to_show + 2].legend()

	pad = 4 - len(str(eval_tracker['evals']))
	zero_padding = '0' * pad
	evals = eval_tracker['evals']

	fig.tight_layout()
	fig.savefig(f'{out_dir}/{zero_padding}{evals}.png')
	plt.close('all')

def weight_change_penalty(weight_deltas, s=30):
	n = len(weight_deltas)
	b = np.exp(1/s)
	a = (1 - b) / (b * (1 - np.power(b, n))) # this normalizes total weight change penalty to 1
	return a * np.power(b, np.arange(1, n + 1))

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

	w_hist = []
	all_weight_deltas = []

	w_hist.append(w)

	for i in range(n_inner_loop_iters):
		# Define input for activation of the network
		r_in = np.zeros((len(t), n_e + n_i))
		input_amp = np.random.rand() * 0.002 + 0.01
		r_in[:, 0] = generate_gaussian_pulse(t, 5e-3, 5e-3, w=input_amp) # Drive first excitatory cell with Gaussian input

		# below, simulate one activation of the network for the period T
		r, s, v, w_out, effects = simulate(t, n_e, n_i, r_in + 4e-6 / dt * np.random.rand(len(t), n_e + n_i), plasticity_coefs, w, w_plastic, dt=dt, tau_e=10e-3, tau_i=0.1e-3, g=1, w_u=1, track_params=track_params)

		loss = l2_loss(r, r_target)
		cumulative_loss = cumulative_loss * gamma + loss

		if np.isnan(r).any(): # if simulation turns up nans in firing rate matrix, end the simulation
			break

		all_weight_deltas.append(np.sum(np.abs(w_out - w_hist[0])))

		w_hist.append(w_out)
		if len(w_hist) > DW_LAG:
			w_hist.pop(0)

		if effects is not None:
			all_effects += effects

		w = w_out # use output weights evolved under plasticity rules to begin the next simulation


	penalty_wc = DW_PENALTY / np.sum(all_weight_deltas) * np.dot(weight_change_penalty(all_weight_deltas), all_weight_deltas)

	if np.isnan(penalty_wc):
		penalty_wc = 0

	print(f'penalty_wc {index}:', penalty_wc)

	normed_loss = cumulative_loss / (1 / np.log(1/gamma)) + penalty_wc

	return r, w, w_initial, normed_loss, all_effects, all_weight_deltas

# Function to minimize (including simulation)

def simulate_plasticity_rules(plasticity_coefs, eval_tracker=None, track_params=False):
	start = time.time()

	pool = mp.Pool(POOL_SIZE)
	f = partial(simulate_single_network, plasticity_coefs=plasticity_coefs, track_params=track_params)
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

# x1 = str('-0.037734288820665436 -0.03011735209391469 -0.018173897036771875 0.03600648076171382 0.0019114866230664235 -0.05101598199784114 0.05896772937739326 0.05547212166030329 -0.02005802902652238 -0.02474866467251842 -0.01993604835992464 -0.034195429167007636 -0.04357875335317038 0.029534815609442443 0.04297850591742626 0.03508071471093647 0.01003288104415306 -0.015620211977487523 -0.03952208297239813 0.1104667187122368 0.02349686679274084 0.02411736376057691 0.034196874600540536 -0.014620788855925155 -0.04259783311340164 -0.03054606723461701 -0.023801372010937166 -0.07402352486400733 -0.05601334479094237 -0.02299363314607309 -0.02515251162699948 0.0015077839488898126 0.10783878933203547 -0.017914736864477344 0.024163036320314784 -0.06828155308142446 0.0011158678601269328 -0.07878881973302962 -0.06321771995274984 0.015011653114717138 -0.15655131899845917 0.04758066510878306 0.04608984375891524 0.08273776999253006 0.1060621707873952 -0.023336389307702796 0.08877043115911275 -0.04379972387400109').split(' ')
# x1 = np.array([float(n) for n in x1])

# x1[8] = 0.005

# effect_sizes = np.array([5.27751763e+05, 6.18235136e+04, 5.21351071e+04, 1.34717956e+04,
#  1.05863156e+03, 3.63087241e+05, 4.24716499e+05, 7.99887936e+04,
#  1.68607385e+05, 2.99179664e+04, 3.35792818e+04, 3.28196266e+04,
#  6.71993169e+04, 1.24480543e+05, 1.86945603e+05, 6.87652031e+04,
#  3.56412481e+05, 1.68045926e+05, 3.30536029e+05, 3.10001561e+05,
#  7.03971418e+04, 9.15983754e+05, 6.24189161e+05, 9.16165296e+04,
#  6.88224954e+05, 1.49140300e+05, 8.99853529e+04, 9.69494421e+04,
#  7.87418244e+04, 3.96323730e+05, 2.09012209e+05, 4.29591886e+03,
#  3.54434827e+06, 8.62402443e+04, 1.63261431e+05, 1.61037923e+05,
#  4.59451709e+03, 1.32908452e+06, 2.19338468e+06, 8.66411375e+04,
#  8.67751299e+05, 3.82004006e+04, 5.59146540e+04, 4.17414558e+04,
#  9.47268052e+04, 7.05007186e+04, 5.89641175e+05, 4.11536245e+04,])

# num_to_silence = [13, 14, 15]
# for j in range(3):
# 	st = 16 * j
# 	en = 16 * (j + 1)
# 	effects_partial = effect_sizes[st:en]
# 	set_smallest_n_zero(effects_partial, num_to_silence[j], arr_set=x1[st:en])

# x1 = copy(x0)
# x1[0] = 1e-2

# simulate_plasticity_rules(x1, eval_tracker=eval_tracker, track_params=True)
# simulate_plasticity_rules(x1, eval_tracker=eval_tracker, track_params=True)
# simulate_plasticity_rules(x1, eval_tracker=eval_tracker, track_params=True)
# simulate_plasticity_rules(x1, eval_tracker=eval_tracker, track_params=True)

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
