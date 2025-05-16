# The main simulation function
"""
This is the simulation function used to run all of the simulations described in
Section 5, as well as the sensitivity analyses presented in Section S8 of the
Supplement. The results of these simulations are shown in Figures 3, 4, 5 in the
main text and Figures S1 through S27 in the Supplement.
""" 

import os
import sys
import time

import numpy as np

from ipss.helpers import compute_q_list, tps_and_fps
from ipss.main import lassocv, mcpcv, scadcv, select
from simulations.generate_data import generate_data

#--------------------------------
# Simulation function
#--------------------------------
def run_simulation(simulation_config):

	start = time.time()

	simulation_name = simulation_config['simulation_name']

	random_seed = simulation_config['random_seed']
	np.random.seed(random_seed)

	response_type = simulation_config['response_type']
	feature_matrix = simulation_config['feature_matrix']
	n_trials = simulation_config['n_trials']
	p = simulation_config['p']
	n_range = simulation_config['n_range']
	snr_range = simulation_config['snr_range']
	p_true_range = simulation_config['p_true_range']
	efp_list = simulation_config['efp_list']
	methods_to_run = simulation_config['methods_to_run']

	df = simulation_config.setdefault('df', False)
	select_args = simulation_config.setdefault('select_args', {})
	ipss_args = simulation_config.setdefault('ipss_args', None)
	ss_args = simulation_config.setdefault('ss_args', None)

	if isinstance(n_range, (int, float)):
		n_range = [n_range, n_range + 1]
	if isinstance(snr_range, (int, float)):
		snr_range = [snr_range, snr_range + 1]
	if isinstance(p_true_range, (int, float)):
		p_true_range = [p_true_range, p_true_range + 1]

	# ipss results dict
	if 'ipss' in methods_to_run:
		function_list = ipss_args['function_list']
		cutoff_list = ipss_args['cutoff_list']
		delta_list = ipss_args['delta_list']
		ipss_results = {function:{cutoff:{delta: {'tp_list': np.zeros_like(efp_list), 'fp_list': np.zeros_like(efp_list)}
			for delta in delta_list} for cutoff in cutoff_list} for function in function_list}

	# ss results dict
	if 'ss' in methods_to_run:
		ss_args = simulation_config['ss_args']
		assumption_list = ss_args['assumption_list']
		tau_list = ss_args['tau_list']
		ss_results = {assumption:{tau: {'tp_list': np.zeros_like(efp_list), 'fp_list': np.zeros_like(efp_list)} 
			for tau in tau_list} for assumption in assumption_list}
		q_list = {assumption: {tau: [] for tau in tau_list} for assumption in assumption_list}
		for assumption in assumption_list:
			for tau in tau_list:
				q_list[assumption][tau] = compute_q_list(efp_list, tau, method=assumption, p=p, B=50)
		ss_args['q_list'] = q_list

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

	runtime = time.time() - start
	print(f'Simulation time: {runtime:.1f} seconds ({runtime/n_trials:.1f} s/trial)')
	print(f'')

	return {'ipss_results': ipss_results, 'ss_results': ss_results, 'lassocv_results':lassocv_results, 
		'mcpcv_results':mcpcv_results, 'scadcv_results':scadcv_results, 'simulation_config': simulation_config}


		