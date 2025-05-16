# Simulation experiments
"""
Run the simulation experiments, as described in Section 5. This is the base code used 
to produce Figures 3, 4, and 5 in the main text and Figures S1 through S25 in the 
Supplement. In particular, this code can be used to run both the primary simulation 
results (Section 5 and S7) as well as the sensitivity analyses (Section S8).

The code is designed to be run once (e.g., 100 trials) for all of the methods and with
different parameter choices (e.g., cutoff and delta values for IPSS and tau values for
the stability selection methods). The results should be saved by setting save_results
to True. These results can then be analyzed for different methods (as in Figures 4 and 
5 in the main text and Figures S2 through S9 in the Supplement), or for different
parameter choices, as done in the sensitivity analysis in Section S8 of the Supplement
(Figures S10 through S25).

Runtime: Running 100 trials takes between 10 minutes and 2 hours depending upon the
dimension, p, and the base selector (lasso, SCAD, MCP, or logistic regression)
""" 

import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from simulations.simulation_function import run_simulation

################################
save_results = False
################################

# Set random seed
random_seed = 302
np.random.seed(random_seed)

#--------------------------------
# Simulation parameters
#--------------------------------
# Feature design
"""
feature_matrix = 0: Standard normal
feature_matrix in (0, 1): Multivariate Gaussian with Toeplitz covariance matrix and correlation strength = feature_matrix
feature_matrix = 'ovarian_cancer' for RNA-seq features from ovarian cancer patients
"""
feature_matrix = 0.5
# import ovarian cancer data if feature_matrix == 'ovarian_cancer'
if feature_matrix == 'ovarian_cancer':
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	data_path = os.path.join(repo_root, 'data', 'ovarian_data.npy')
	data = np.load(data_path, allow_pickle=True).item()

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
if response_type == 'linear_class':
	if 'mcpcv' in methods_to_run or 'scadcv' in methods_to_run:
		print(f'Warning: MCPCV and SCADCV are not compatible with logistic regression.')

# Base estimator
"""
Choices for selector: 
	- 'l1' for lasso or logistic regression 
	- 'adaptive' for adaptive lasso
	- 'mcp' for minimax concave penalty (MCP)
	- 'scad' for smoothly clipped aboslute deviation (SCAD)
"""
selector = 'l1'

# Simulation name
if isinstance(feature_matrix, (int, float)):
	if feature_matrix == 0:
		simulation_name = f'independent_{response_type}_{p}'
	elif np.isscalar(feature_matrix) and 0 < feature_matrix <= 1:
		simulation_name = f'toeplitz_{feature_matrix}_{response_type}_{p}'
else:
	simulation_name = f'oc_rnaseq_{response_type}_{p}'
if df:
	simulation_name += f'_df{df}'
if selector == 'adaptive':
	simulation_name += f'_adaptive'
if selector == 'mcp':
	simulation_name += f'_mcp'
if selector == 'scad':
	simulation_name += f'_scad'

# Store simulation details
simulation_config = {
	'simulation_name':simulation_name,
	'response_type':response_type,
	'feature_matrix':feature_matrix,
	'n_trials':n_trials,
	'p':p,
	'n_range':n_range,
	'p_true_range':p_true_range,
	'snr_range':snr_range,
	'efp_list':efp_list,
	'random_seed':random_seed,
	'methods_to_run':methods_to_run
}

#--------------------------------
# Methods specifications
#--------------------------------
if 'ipss' in methods_to_run or 'ss' in methods_to_run:
	select_args = {'B':50, 'n_alphas':25, 'selector':selector}

	# ipss args
	ipss_args = {
		'function_list': ['h2', 'h3'],
		'cutoff_list': [0.025, 0.05, 0.075, 0.1],
		'delta_list': [0, 1/4, 1/2, 3/4, 1, 5/4, 3/2],
		}
	simulation_config['ipss_args'] = ipss_args

	# stability selection
	assumption_list = ['none', 'unimodal', 'r-concave']
	tau_list = [0.6, 0.75, 0.9]
	ss_args = {
		'assumption_list': assumption_list,
		'efp_list': efp_list,
		'tau_list': tau_list
		}
	simulation_config['ss_args'] = ss_args

#--------------------------------
# Run simulation
#--------------------------------
results = run_simulation(simulation_config)

simulation_config = results['simulation_config']
efp_list = simulation_config['efp_list']

if save_results:
	file_name = simulation_name + '.pkl'
	with open(file_name, 'wb') as f:
		pickle.dump(results, f)

#--------------------------------
# Plot results
#--------------------------------
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








