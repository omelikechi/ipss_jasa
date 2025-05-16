# Code to reproduce Figure 1 in "Integrated path stability selection"
"""
This figure shows simulation results for lasso with cross-validation (LassoCV) and
the stability selection methods (MB, UM, and r-concave). The simulation design consists
of n = 200 samples, p = 1000 features drawn from a standard normal distribution, and
a linear response y = Xb + epsilon where b has 20 nonzero entries and epsilon is
Gaussian with variance chosen so that the signal-to-noise ratio is 2.

Runtime: Running 100 trials takes about 30 minutes on a MacBook Pro
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

from simulations.simulation_function import run_simulation

# Set random seed
random_seed = 302
np.random.seed(random_seed)

#--------------------------------
# Simulation parameters
#--------------------------------
response_type = 'linear_reg'
feature_matrix = 0 # standard normal features
n_trials = 100
p = 1000
n = 200
p_true = 20
snr = 2
efp_list = np.linspace(0, 5, 21)
methods_to_run = ['ss', 'lassocv']
simulation_name = 'figure1'

# Store simulation details
simulation_config = {
	'response_type':response_type,
	'feature_matrix':feature_matrix,
	'n_trials':n_trials,
	'p':p,
	'n_range':n,
	'p_true_range':p_true,
	'snr_range':snr,
	'efp_list':efp_list,
	'methods_to_run':methods_to_run,
	'random_seed':random_seed,
	'simulation_name':simulation_name
}

#--------------------------------
# Method specifications
#--------------------------------
# select args
select_args = {'n_alphas':100}
simulation_config['select_args'] = select_args

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

file_name = simulation_name + '_results.pkl'
with open(file_name, 'wb') as f:
	pickle.dump(results, f)

simulation_config = results['simulation_config']
efp_list = simulation_config['efp_list']

#--------------------------------
# Plot results
#--------------------------------
# plot specifications
fig, ax = plt.subplots(1, 2, figsize=(16,6))

linewidth = 3.5

# stability selection results
assumption_colors = {'none':'darkorchid', 'unimodal':'deepskyblue', 'r-concave':'limegreen'}
assumption_names = {'none':'MB', 'unimodal':'UM', 'r-concave':'r-concave'}
tau_style = {0.6:'-', 0.75:':', 0.9:'--'}

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
			ax[0].plot(efp_list, fps, label=assumption_names[assumption], color=color, linestyle=linestyle, lw=linewidth)
			ax[1].plot(efp_list, tps, label=assumption_names[assumption], color=color, linestyle=linestyle, lw=linewidth)
		else:
			ax[0].plot(efp_list, fps, color=color, linestyle=linestyle, lw=linewidth)
			ax[1].plot(efp_list, tps, color=color, linestyle=linestyle, lw=linewidth)

# lassocv results
color = 'gold'
linestyle = '-'
lassocv_results = results['lassocv_results']
fp = lassocv_results['fp']
tp = lassocv_results['tp']
ax[0].axhline(fp, color=color, label='lassoCV', linestyle=linestyle, lw=linewidth)
ax[1].axhline(tp, color=color, label='lassoCV', linestyle=linestyle, lw=linewidth)

# plot target E(FP)
ax[0].plot(efp_list, efp_list, color='black', linestyle='--', lw=linewidth)

ax[0].grid(True)
ax[1].grid(True)

ax[0].set_ylabel('Average FP', fontsize=24)
ax[0].set_xlabel('Target E(FP)', fontsize=24)
ax[1].set_ylabel('Average TP', fontsize=24)
ax[1].set_xlabel('Target E(FP)', fontsize=24)

# ax[1].legend(loc='best', fontsize=24)
legend = ax[1].legend(loc='upper right', fontsize=14)
for legline in legend.get_lines():
	legline.set_linewidth(5)

plt.tight_layout()
plt.savefig('figure1.png', dpi=300)
plt.show()

print(f'')











