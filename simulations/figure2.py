# Code to reproduce Figure 2 in "Integrated path stability selection"
"""
This figure shows stability paths and results from the IPSS and stability selection
methods. The simulation design consists of n = 150 samples, p = 200 features drawn 
from a standard normal distribution, and a linear response y = Xb + epsilon where b 
has 15 nonzero entries and epsilon is Gaussian with variance chosen so that the 
signal-to-noise ratio is 1.

Runtime: About 10 seconds on a MacBook Pro
"""

import time

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from helpers import compute_efp_scores, stabsel_q
from main import compute_stability_paths
from simulations.generate_data import generate_data

# Set random seed
random_seed = 2
np.random.seed(random_seed)

#--------------------------------
# Simulation parameters
#--------------------------------
# data parameters
response_type = 'linear_reg'
feature_matrix = 0 # standard normal features
n_trials = 10
p = 200
n = 150
p_true = 15
snr = 1

# stability selection parameters
B = 50
efp = 1

#--------------------------------
# Run simulation
#--------------------------------
X, y, true_features = generate_data(n, p, p_true, snr, response_type, feature_matrix, random_seed=random_seed)
path_results = compute_stability_paths(X, y, n_alphas=100)
alphas = path_results['alphas']
average_selected = path_results['average_selected']
stability_paths = path_results['stability_paths']

#--------------------------------
# Results for different methods
#--------------------------------
method_list = ['IPSS(cubic)', 'IPSS(quad)', 'r-concave', 'UM', 'MB']
method_colors = {'IPSS(cubic)':'red', 'IPSS(quad)':'orange', 'r-concave':'limegreen', 'UM':'deepskyblue', 'MB':'darkorchid'}
method_scores = {}
method_stop = {}
method_threshold = {}

# ipss(cubic) results
ipss_function = 'h3'
delta = 1
cutoff = 0.05
scores, integral, _, stop_index = compute_efp_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff)
method_scores['IPSS(cubic)'] = scores
method_stop['IPSS(cubic)'] = stop_index
method_threshold['IPSS(cubic)'] = integral

# ipss(quad) results
ipss_function = 'h2'
delta = 1
cutoff = 0.05
scores, integral, _, stop_index = compute_efp_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff)
method_scores['IPSS(quad)'] = scores
method_stop['IPSS(quad)'] = stop_index
method_threshold['IPSS(quad)'] = integral

# stability selection results
cutoff = 0.75

# r-concave
q = stabsel_q(p, cutoff=cutoff, PFER=efp, B=B, assumption='r-concave')
stop_index = np.where(average_selected <= q)[0].max()
method_threshold['r-concave'] = cutoff
method_stop['r-concave'] = stop_index
method_scores['r-concave'] = np.max(stability_paths[:stop_index, :], axis=0)

# UM
q = stabsel_q(p, cutoff=cutoff, PFER=efp, B=B, assumption='unimodal')
stop_index = np.where(average_selected <= q)[0].max()
method_threshold['UM'] = cutoff
method_stop['UM'] = stop_index
method_scores['UM'] = np.max(stability_paths[:stop_index, :], axis=0)

# MB
q = stabsel_q(p, cutoff=cutoff, PFER=efp, B=B, assumption='none')
stop_index = np.where(average_selected <= q)[0].max()
method_threshold['MB'] = cutoff
method_stop['MB'] = stop_index
method_scores['MB'] = np.max(stability_paths[:stop_index, :], axis=0)

#--------------------------------
# Plot results
#--------------------------------
# plot specifications
colors = ['dodgerblue' if j in true_features else 'gray' for j in np.arange(p)]
edgecolors = ['black' if i in true_features else None for i in range(p)]
transparency = [1 if j in true_features else 0.75 for j in range(p)]
linewidths = [2 if j in true_features else 1 for j in range(p)]

# stability paths
stability_paths = stability_paths[:method_stop['IPSS(cubic)'] + 1, :]
n_alphas, p = stability_paths.shape

# plot the scores for each method
fig, ax = plt.subplots(len(method_list), 1, figsize=(14, 8))

for i, method in enumerate(method_list):
	threshold = method_threshold[method]
	scores = method_scores[method]
	noise = np.random.uniform(-0.1, 0.1, size=p)
	for j in range(p):
		ax[i].scatter(scores[j], noise[j], color=colors[j], s=150, alpha=transparency[j], edgecolor=edgecolors[j])
	ax[i].axvline(threshold, color='red', linestyle='--', lw=3)
	ax[i].set_yticks([])
	ax[i].set_ylabel(method, fontsize=14)

plt.tight_layout()
plt.show()

# plot the stability paths
plt.figure(figsize=(12, 7))

for j in range(p):
	plt.plot(np.arange(n_alphas), stability_paths[:n_alphas, j], color=colors[j], alpha=transparency[j], lw=linewidths[j])


for method in method_list:
	plt.axvline(method_stop[method], linestyle='--', lw=3, color=method_colors[method], label=method)

plt.xticks([])

# custom legend labels (solid line rather than dashed)
legend_handles = [mlines.Line2D([], [], color=method_colors[method], lw=3, linestyle='-', label=method) for method in method_list]
plt.legend(handles=legend_handles, loc='best')

plt.tight_layout()
plt.show()











