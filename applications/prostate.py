# Analyze colon cancer data

import matplotlib.pyplot as plt
import numpy as np
import pickle

from helpers import compute_q_list, tps_and_fps
from main import lassocv, mcpcv, scadcv, select

# load data
data = np.load('./applications/prostate_data.npy', allow_pickle=True).item()
X, y, feature_names = data['features'], data['response'], data['feature_names']
n, p = X.shape

# set random seed
random_seed = 14
np.random.seed(random_seed)

# E(FP) specs
target_efp = 1
efp_list = np.linspace(0, 5, 21)

# select args
select_args = {'B':50, 'n_alphas':25, 'selector':'l1'}

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

for function in ipss_args['function_list']:
	for cutoff in ipss_args['cutoff_list']:
		for delta in ipss_args['delta_list']:
			scores = ipss_scores[function][cutoff][delta]
			total_selected = 0
			print(f'Proteins selected by IPSS({function}):')
			for j in scores.keys():
				if scores[j] <= target_efp:
					total_selected += 1
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
					print(f'  {feature_names[j]}')
		print(f'Total number selected: {total_selected}')

	print()













