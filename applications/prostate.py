# Analyze prostate cancer data
"""
The results are reported in Section 6.1. Here, we provide code for selecting the proteins related
to tumor purity and for plotting the associated stability paths. Note that some of the proteins 
selected by each method can change with the random seed; however, most of the proteins selected by 
the different methods are consistent across different random seeds.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ipss.helpers import compute_q_list, tps_and_fps
from ipss.main import lassocv, mcpcv, scadcv, select

# load data
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(repo_root, 'data', 'prostate_data.npy')
data = np.load(data_path, allow_pickle=True).item()
X, y, feature_names = data['features'], data['response'], data['feature_names']
n, p = X.shape

# set random seed
random_seed = 8
np.random.seed(random_seed)

# E(FP) specs
target_efp = 1
efp_list = np.linspace(0, 5, 21)

# select args
select_args = {'B':50, 'n_alphas':100}

# ipssl args
ipss_args = {
	'function_list': ['h2', 'h3'],
	'cutoff_list': [0.05],
	'delta_list': [1],
	}

# stability selection
assumption_list = ['none', 'unimodal', 'r-concave']
assumption_name = {'none':'MB', 'unimodal':'UM', 'r-concave':'r-concave'}
tau_list = [0.75]
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

results = select(X, y, ipss_args, ss_args, **select_args)
ipss_scores = results['ipss_efp_scores_list']
ss_scores = results['ss_efp_scores_list']

selected_proteins = {}

for function in ipss_args['function_list']:
	for cutoff in ipss_args['cutoff_list']:
		for delta in ipss_args['delta_list']:
			scores = ipss_scores[function][cutoff][delta]
			total_selected = 0
			print(f'Proteins selected by IPSS({function}):')
			for j in scores.keys():
				if scores[j] <= target_efp:
					total_selected += 1
					selected_proteins[j] = feature_names[j]
					print(f'  {feature_names[j]}')
			print(f'Total number selected: {total_selected}')
	print()

for assumption in ss_args['assumption_list']:
	for tau in ss_args['tau_list']:
		scores = ss_scores[assumption][tau]
		total_selected = 0
		print(f'Proteins selected by SS({assumption_name[assumption]}):')
		for j in scores.keys():
				if scores[j] <= target_efp:
					total_selected += 1
					selected_proteins[j] = feature_names[j]
					print(f'  {feature_names[j]}')
		print(f'Total number selected: {total_selected}')
	print()

# plot stability paths
stability_paths = results['stability_paths']

n_alphas, p = stability_paths.shape

# cut stability paths short for better visualization
n_alphas = int(2 * n_alphas / 3)

def generate_rainbow_colors(n):
	colors = plt.cm.hsv(np.linspace(0, 1, n))
	np.random.shuffle(colors)
	return colors

colors = generate_rainbow_colors(len(selected_proteins))
lws = [3 if j in selected_proteins else 1.5 for j in range(p)]

color_idx = 0
for j in range(p):
	if j in selected_proteins.keys():
		plt.plot(np.arange(n_alphas), stability_paths[:n_alphas,j], color=colors[color_idx], label=selected_proteins[j], lw=lws[j])
		color_idx += 1
	else:
		plt.plot(np.arange(n_alphas), stability_paths[:n_alphas,j], color='gray', lw=lws[j])

plt.xticks([])
plt.xlabel(f'$-\\log(\\lambda)$', fontsize=18)

plt.legend(loc='best')
plt.tight_layout()
plt.show()












