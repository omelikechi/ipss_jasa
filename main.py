# Implement IPSS and stability selection methods
"""
The main function is select.
"""

import time
import warnings

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from skglm.estimators import GeneralizedLinearEstimator, MCPRegression
from skglm.penalties import SCAD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, lasso_path, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from helpers import (check_response_type, compute_alphas, compute_efp_scores, compute_q_list, compute_qvalues, integrate, 
	score_based_selection, selection, selector_and_args)

#--------------------------------
# Main function
#--------------------------------
def select(X, y, ipss_args, ss_args, B=50, selector='l1', selector_args=None, n_alphas=None, standardize_X=None, center_y=None, n_jobs=1):

	# start timer
	start = time.time()

	p_full = X.shape[1]

	if selector in ['l1', 'mcp', 'scad']:
		selector_type = 'regularization'
	else:
		selector_type = 'thresholding'

	output = compute_stability_paths(X, y, selector_type=selector_type, selector=selector, selector_args=selector_args, 
		B=B, n_alphas=n_alphas, standardize_X=standardize_X, center_y=center_y, n_jobs=n_jobs)

	stability_paths = output['stability_paths']
	alphas = output['alphas']
	average_selected = output['average_selected']

	n_alphas, p = stability_paths.shape

	# ipss
	function_list = ipss_args['function_list']
	cutoff_list = ipss_args['cutoff_list']
	delta_list = ipss_args['delta_list']
	ipss_efp_scores_list = {function:{cutoff:{delta: [] for delta in delta_list} for cutoff in cutoff_list} for function in function_list}

	for function in function_list:
		for cutoff in cutoff_list:
			for delta in delta_list:
				scores, integral, alphas, stop_index = compute_efp_scores(stability_paths, B, alphas, average_selected, function, delta, cutoff)
				efp_scores = np.round(integral / np.maximum(scores, integral / p), decimals=8)
				efp_scores = dict(zip(np.arange(p), efp_scores))
				ipss_efp_scores_list[function][cutoff][delta] = efp_scores

	# stability selection
	assumption_list = ss_args['assumption_list']
	efp_list = ss_args['efp_list']
	tau_list = ss_args['tau_list']
	q_list = ss_args['q_list']
	ss_efp_scores_list = {assumption: {tau: [] for tau in tau_list} for assumption in assumption_list}
	for assumption in assumption_list:
		for tau in tau_list:
			efp_scores = p * np.ones(p)
			already_selected = []
			for idx, efp in enumerate(efp_list):
				if efp == 0:
					continue
				q = q_list[assumption][tau][idx]
				stop_index = max(np.argmin(np.abs(average_selected - q)), 1)
				stop_index = min(stop_index, n_alphas)
				stability_scores = np.max(stability_paths[:stop_index, :], axis=0)
				for j in range(p):
					if stability_scores[j] >= tau and j not in already_selected:
						efp_scores[j] = efp
						already_selected.append(j)

			efp_scores = dict(zip(np.arange(p), efp_scores))
			ss_efp_scores_list[assumption][tau] = efp_scores

	runtime = time.time() - start

	return { 
		'ipss_efp_scores_list': ipss_efp_scores_list,
		'ss_efp_scores_list': ss_efp_scores_list,
		'runtime': runtime, 
		'stability_paths': stability_paths
		}

#--------------------------------
# Other methods
#--------------------------------
def cross_validate(model, X, y, alpha_range, cv=5):
	kf = KFold(n_splits=cv, shuffle=True, random_state=302)
	mean_errors = []
	for alpha in alpha_range:
		errors = []
		for train_idx, val_idx in kf.split(X):
			X_train, X_val = X[train_idx], X[val_idx]
			y_train, y_val = y[train_idx], y[val_idx]
			# Fit model on the training data
			clf = model(alpha=alpha).fit(X_train, y_train)
			# Predict and compute mean squared error
			y_pred = clf.predict(X_val)
			errors.append(np.mean((y_val - y_pred) ** 2))
		mean_errors.append(np.mean(errors))
	best_alpha = alpha_range[np.argmin(mean_errors)]
	return best_alpha

def lassocv(X, y, true_features):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		lasso = LassoCV().fit(X, y)
	selected_features = np.where(lasso.coef_ != 0)[0]
	tp, fp = 0, 0
	for feature in selected_features:
		if feature in true_features:
			tp += 1
		else:
			fp += 1
	return tp, fp

def mcpcv(X, y, true_features, gamma=3.0, cv=5):
	n = X.shape[0]
	alpha_max = np.max(np.abs(X.T @ y)) / n
	alpha_range = np.logspace(np.log10(alpha_max/100), 2 * np.log10(alpha_max), 25)
	def model(alpha):
		return MCPRegression(alpha=alpha, gamma=gamma)
	best_alpha = cross_validate(model, X, y, alpha_range, cv=cv)
	mcp_model = model(best_alpha).fit(X, y)
	selected_features = np.where(mcp_model.coef_ != 0)[0]
	tp, fp = 0, 0
	for feature in selected_features:
		if feature in true_features:
			tp += 1
		else:
			fp += 1
	return tp, fp

def scadcv(X, y, true_features, gamma=3.7, cv=5):
	n = X.shape[0]
	alpha_max = np.max(np.abs(X.T @ y)) / n
	alpha_range = np.logspace(np.log10(alpha_max/100), np.log10(alpha_max), 25)

	def model(alpha):
		penalty = SCAD(alpha=alpha, gamma=gamma)
		return GeneralizedLinearEstimator(penalty=penalty)

	# Perform cross-validation
	best_alpha = cross_validate(model, X, y, alpha_range, cv=cv)

	# Fit the final model with the best alpha
	scad_model = model(best_alpha).fit(X, y)

	# Compute true positives and false positives
	selected_features = np.where(scad_model.coef_ != 0)[0]
	tp, fp = 0, 0
	for feature in selected_features:
		if feature in true_features:
			tp += 1
		else:
			fp += 1

	return tp, fp

#--------------------------------
# Compute stability paths
#--------------------------------
def compute_stability_paths(X, y, selector_type='thresholding', selector='gb', selector_args=None, 
	B=None, n_alphas=None, standardize_X=None, center_y=None, n_jobs=1):

	# empty set for selector args if none specified
	selector_args = selector_args or {}

	# number of subsamples
	B = B if B is not None else 100 if selector == 'gb' else 50

	# reshape response
	if len(y.shape) > 1:
		y = y.ravel()
	
	# check response type
	binary_response, selector = check_response_type(y, selector)

	# standardize and center data if using l1 selectors
	if selector_type == 'regularization':
		if standardize_X is None:
			X = StandardScaler().fit_transform(X)
		if center_y is None:
			if not binary_response:
				y -= np.mean(y)
	
	# dimensions
	n, p = X.shape
	
	# maximum number of features for l1 regularized selectors (to avoid computational issues)
	max_features = 0.75 * p if selector_type == 'regularization' else None

	# alphas
	if n_alphas is None:
		n_alphas = 25 if selector_type == 'regularization' else 100
	alphas = compute_alphas(X, y, n_alphas, max_features, binary_response) if selector_type == 'regularization' else None

	# selector function and args
	selector_function, selector_args = selector_and_args(selector, selector_args)

	# estimate selection probabilities
	results = np.array(Parallel(n_jobs=n_jobs)(delayed(selection)(X, y, alphas, selector_function, **selector_args) for _ in range(B)))

	# score-based selection
	if alphas is None:
		results, alphas = score_based_selection(results, n_alphas)

	# aggregate results
	Z = np.zeros((n_alphas, 2*B, p))
	for b in range(B):
		Z[:, 2*b:2*(b + 1), :] = results[b,:,:,:]

	# average number of features selected (denoted q in ipss papers)
	average_selected = np.array([np.mean(np.sum(Z[i,:,:], axis=1)) for i in range(n_alphas)])

	# stability paths
	stability_paths = np.empty((n_alphas,p))
	for i in range(n_alphas):
		stability_paths[i] = Z[i].mean(axis=0)

	# stop if all stability paths stop changing (after burn-in period where mean selection probability < 0.01)
	stop_index = n_alphas
	for i in range(2,len(alphas)):
		if np.isclose(stability_paths[i,:], np.zeros(p)).all():
			continue
		else:
			diff = stability_paths[i,:] - stability_paths[i-2,:]
			mean = np.mean(stability_paths[i,:])
			if np.isclose(diff, np.zeros(p)).all() and mean > 0.01:
				stop_index = i
				break

	# truncate stability paths at stop index
	stability_paths = stability_paths[:stop_index,:]
	alphas = alphas[:stop_index]
	average_selected = average_selected[:stop_index]

	return {'stability_paths':stability_paths, 'average_selected':average_selected, 'alphas':alphas}



