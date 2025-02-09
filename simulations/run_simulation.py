# Simulation experiments
"""
Run the simulation experiments, as described in Section 5 of "Integrated path stability selection"
""" 

import time

import matplotlib.pyplot as plt
import numpy as np
import pickle

from helpers import compute_q_list, tps_and_fps
from main import lassocv, mcpcv, scadcv, select
from simulations.generate_data import generate_data

#--------------------------------
# Simulation parameters
#--------------------------------

plot_stability_paths = False

# Set random seed
random_seed = 302
np.random.seed(random_seed)

# Feature design
"""
feature_matrix = 0: Standard normal
feature_matrix in (0, 1): Multivariate Gaussian with Toeplitz covariance matrix and correlation strength = feature_matrix
feature_matrix = np.load('ovarian_rnaseq.npy') for RNA-seq features from ovarian cancer patients
"""
feature_matrix = 0.5

# Response type
"""
response_type = 'linear_reg' for linear regression
response_type = 'linear_class' for logistic regression (binary response)
"""
response_type = 'linear_reg'

# Degrees of freedom for Student's t noise (heavy tails); False for Gaussian noise
df = False

# Number of trials to run
n_trials = 10

# Simulation parameters
"""
p = number of features
n_range = range for number of samples
p_true_range = range for number of true (non-null) features
snr_range = range for signal-to-noise ratio
"""
p = 200

# auto select other default parameters based on p
n_range = [50, 200] if p == 200 else [100, 500]
p_true_range = [5, 20] if p == 200 else [10, 40]
snr_range = [1/3, 3] if response_type == 'linear_reg' else [1/2, 2]

# List of target E(FP) values
efp_list = np.linspace(0, 5, 21)

# Methods to run
methods_to_run = ['ipss', 'ss', 'lassocv', 'mcpcv', 'scadcv']

# Base estimator
"""
Choices: 'l1' (for lasso), 'mcp', or 'scad'
"""
selector = 'l1'

# Simulation name
if feature_matrix == 0:
	simulation_name = f'independent_{response_type}_{p}'
elif np.isscalar(feature_matrix) and 0 < feature_matrix <= 1:
	simulation_name = f'toeplitz_{feature_matrix}_{response_type}_{p}'
else:
	simulation_name = f'oc_rnaseq_{response_type}_{p}'
if df:
	simulation_name += f'_df{df}'
if selector == 'mcp':
	simulation_name += f'_mcp'
if selector == 'scad':
	simulation_name += f'_scad'

# Store simulation details
simulation_config = {
	'simulation_name': simulation_name,
	'response_type': response_type,
	'feature_matrix': feature_matrix,
	'n_trials': n_trials,
	'p': p,
	'n_range': n_range,
	'p_true_range': p_true_range,
	'snr_range': snr_range,
	'efp_list': efp_list,
	'random_seed': random_seed,
}

#--------------------------------
# Methods and args
#--------------------------------
if 'ipss' in methods_to_run or 'ss' in methods_to_run:
	# selection probability args
	select_args = {'B':50, 'n_alphas':25, 'selector':selector}

	# ipssl args
	ipss_args = {
		'function_list': ['h2', 'h3'],
		'cutoff_list': [0.025, 0.05, 0.075, 0.1],
		'delta_list': [0, 1/4, 1/2, 3/4, 1, 5/4, 3/2],
		}
	simulation_config['ipss_args'] = ipss_args

	# stability selection
	assumption_list = ['none', 'unimodal', 'r-concave']
	tau_list = [0.6, 0.75, 0.9]
	q_list = {assumption: {tau: [] for tau in tau_list} for assumption in assumption_list}
	for assumption in assumption_list:
		for tau in tau_list:
			q_list[assumption][tau] = compute_q_list(efp_list, tau, method=assumption, p=p, B=select_args['B'])

	ss_args = {
		'assumption_list': assumption_list,
		'efp_list': efp_list,
		'q_list': q_list,
		'tau_list': tau_list
		}
	simulation_config['ss_args'] = ss_args

#--------------------------------
# Simulation function
#--------------------------------
def run_simulation(simulation_config):

	start = time.time()

	simulation_name = simulation_config['simulation_name']

	response_type = simulation_config['response_type']
	feature_matrix = simulation_config['feature_matrix']
	n_trials = simulation_config['n_trials']
	p = simulation_config['p']
	n_range = simulation_config['n_range']
	snr_range = simulation_config['snr_range']
	p_true_range = simulation_config['p_true_range']
	efp_list = simulation_config['efp_list']
	random_seed = simulation_config['random_seed']

	# ipss results dict
	if 'ipss' in methods_to_run:
		function_list = ipss_args['function_list']
		cutoff_list = ipss_args['cutoff_list']
		delta_list = ipss_args['delta_list']
		ipss_results = {function:{cutoff:{delta: {'tp_list': np.zeros_like(efp_list), 'fp_list': np.zeros_like(efp_list)}
			for delta in delta_list} for cutoff in cutoff_list} for function in function_list}

	# ss results dict
	if 'ss' in methods_to_run:
		assumption_list = ss_args['assumption_list']
		tau_list = ss_args['tau_list']
		ss_results = {assumption:{tau: {'tp_list': np.zeros_like(efp_list), 'fp_list': np.zeros_like(efp_list)} 
			for tau in tau_list} for assumption in assumption_list}

	# lassocv results
	if 'lassocv' in methods_to_run:
		lassocv_results = {'tp': 0, 'fp': 0}

	# mcp results
	if 'mcpcv' in methods_to_run:
		mcpcv_results = {'tp': 0, 'fp': 0}

	# scad results
	if 'scadcv' in methods_to_run:
		scadcv_results = {'tp': 0, 'fp': 0}

	for trial in range(n_trials):

		print(f'trial {trial + 1}/{n_trials}')

		# Update random seed
		trial_seed = random_seed + trial

		# choose parameters
		n = np.random.randint(n_range[0], n_range[1] + 1)
		p_true = np.random.randint(p_true_range[0], p_true_range[1] + 1)
		snr = np.random.uniform(snr_range[0], snr_range[1])

		# Generate data
		X, y, true_features = generate_data(n, p, p_true, snr, response_type, feature_matrix, df=df, random_seed=trial_seed)

		# lassocv
		if 'lassocv' in methods_to_run:
			tp, fp = lassocv(X, y, true_features)
			lassocv_results['fp'] += fp / n_trials
			lassocv_results['tp'] += tp / n_trials
		else:
			lassocv_results = {}

		# mcpcv
		if 'mcpcv' in methods_to_run:
			tp, fp = mcpcv(X, y, true_features)
			mcpcv_results['fp'] += fp / n_trials
			mcpcv_results['tp'] += tp / n_trials
		else:
			mcpcv_results = {}

		# scadcv
		if 'scadcv' in methods_to_run:
			tp, fp = scadcv(X, y, true_features)
			scadcv_results['fp'] += fp / n_trials
			scadcv_results['tp'] += tp / n_trials
		else:
			scadcv_results = {}

		# run ipss and ss methods
		if 'ipss' in methods_to_run or 'ss' in methods_to_run:
			results = select(X, y, ipss_args, ss_args, **select_args)
			ipss_scores = results['ipss_efp_scores_list']
			ss_scores = results['ss_efp_scores_list']

			if plot_stability_paths:
				import matplotlib.pyplot as plt
				color = ['dodgerblue' if j in true_features else 'gray' for j in np.arange(X.shape[1])]
				stability_paths = results['stability_paths']
				n_alphas, p = stability_paths.shape
				for j in range(p):
					plt.plot(stability_paths[:,j], color=color[j])
				plt.tight_layout()
				plt.show()

		# ipss results
		if 'ipss' in methods_to_run:
			for function in function_list:
				for cutoff in cutoff_list:
					for delta in delta_list:
						efp_scores = ipss_scores[function][cutoff][delta]
						tps, fps = tps_and_fps(efp_scores, efp_list, true_features)
						ipss_results[function][cutoff][delta]['tp_list'] += tps / n_trials
						ipss_results[function][cutoff][delta]['fp_list'] += fps / n_trials
		else:
			ipss_results = {}

		# ss results
		if 'ss' in methods_to_run:
			for assumption in assumption_list:
				for tau in tau_list:
					efp_scores = ss_scores[assumption][tau]
					tps, fps = tps_and_fps(efp_scores, efp_list, true_features)
					ss_results[assumption][tau]['tp_list'] += tps / n_trials
					ss_results[assumption][tau]['fp_list'] += fps / n_trials
		else:
			ss_results = {}

	end = time.time()
	print(f'The simulation took {end - start:.1f} seconds')
	print(f'')

	return {'ipss_results': ipss_results, 'ss_results': ss_results, 'lassocv_results':lassocv_results, 
		'mcpcv_results':mcpcv_results, 'scadcv_results':scadcv_results, 'simulation_config': simulation_config}

#--------------------------------
# Run simulation
#--------------------------------
results = run_simulation(simulation_config)

simulation_config = results['simulation_config']
efp_list = simulation_config['efp_list']

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# lassocv results
if 'lassocv' in methods_to_run:
	color = 'gold'
	linestyle = '-'
	lassocv_results = results['lassocv_results']
	fp = lassocv_results['fp']
	tp = lassocv_results['tp']
	ax[0].axhline(fp, color=color, linestyle=linestyle, lw=2)
	ax[1].axhline(tp, color=color, linestyle=linestyle, lw=2)

# mcpcv results
if 'mcpcv' in methods_to_run:
	color = 'violet'
	linestyle = '-'
	mcpcv_results = results['mcpcv_results']
	fp = mcpcv_results['fp']
	tp = mcpcv_results['tp']
	ax[0].axhline(fp, color=color, linestyle=linestyle, lw=2)
	ax[1].axhline(tp, color=color, linestyle=linestyle, lw=2)

# scadcv results
if 'scadcv' in methods_to_run:
	color = 'tan'
	linestyle = '-'
	scadcv_results = results['scadcv_results']
	fp = scadcv_results['fp']
	tp = scadcv_results['tp']
	ax[0].axhline(fp, color=color, linestyle=linestyle, lw=2)
	ax[1].axhline(tp, color=color, linestyle=linestyle, lw=2)

# ipss results
if 'ipss' in methods_to_run:
	function_colors = {'h2':'orange', 'h3':'red'}
	function_names = {'h2':'IPSS(quad)', 'h3':'IPSS(cubic)'}
	ipss_results = results['ipss_results']
	functions_to_plot = ['h2', 'h3']
	cutoffs_to_plot = [0.05]
	deltas_to_plot = [0 if response_type == 'linear_class' else 3/4 if p == 1000 else 1]
	for function in functions_to_plot:
		for cutoff in cutoffs_to_plot:
			for delta in deltas_to_plot:
				fps = ipss_results[function][cutoff][delta]['fp_list']
				tps = ipss_results[function][cutoff][delta]['tp_list']
				color = function_colors[function]
				ax[0].plot(efp_list, fps, label=function_names[function], color=color, linestyle=linestyle, lw=2)
				ax[1].plot(efp_list, tps, label=function_names[function], color=color, linestyle=linestyle, lw=2)

# stability selection results
if 'ss' in methods_to_run:
	assumption_colors = {'none':'darkorchid', 'unimodal':'deepskyblue', 'r-concave':'limegreen'}
	assumption_names = {'none':'MB', 'unimodal':'UM', 'r-concave':'r-concave'}
	tau_style = {0.6:'-', 0.75:'--', 0.9:':'}

	ss_results = results['ss_results']
	assumptions_to_plot = ss_args['assumption_list']
	taus_to_plot = [0.6, 0.75, 0.9]
	for assumption in assumptions_to_plot:
		for tau in taus_to_plot:
			fps = ss_results[assumption][tau]['fp_list']
			tps = ss_results[assumption][tau]['tp_list']

			color = assumption_colors[assumption]
			linestyle = tau_style[tau]

			if tau == 0.6:
				ax[0].plot(efp_list, fps, label=assumption_names[assumption], color=color, linestyle=linestyle, lw=2)
				ax[1].plot(efp_list, tps, label=assumption_names[assumption], color=color, linestyle=linestyle, lw=2)
			else:
				ax[0].plot(efp_list, fps, color=color, linestyle=linestyle, lw=2)
				ax[1].plot(efp_list, tps, color=color, linestyle=linestyle, lw=2)

ax[0].plot(efp_list, efp_list, color='black', linestyle='--', lw=2)
ax[0].legend(loc='best')

plt.tight_layout()
plt.show()

print(f'')








