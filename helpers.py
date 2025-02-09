# Helper functions for IPSS

import warnings

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector, BoolVector, NULL
stabs = importr('stabs')

from base_selectors import fit_gb_classifier, fit_gb_regressor
from base_selectors import fit_l1_classifier, fit_l1_regressor
from base_selectors import fit_rf_classifier, fit_rf_regressor 
from base_selectors import fit_mcp_regressor, fit_scad_regressor

def check_response_type(y, selector):
	unique_values = np.unique(y)
	if len(unique_values) == 1:
		print(f"Error: The response variable `y` has only one unique value: {unique_values[0]}.")
		return None, None
	binary_response = len(unique_values) == 2
	if binary_response:
		minval = np.min(unique_values)
		y = np.where(y == minval, 0, 1)
	if selector == 'l1':
		selector = 'logistic_regression' if binary_response else 'lasso'
	elif selector == 'rf':
		selector = 'rf_classifier' if binary_response else 'rf_regressor'
	elif selector == 'gb':
		selector = 'gb_classifier' if binary_response else 'gb_regressor'
	return binary_response, selector

def compute_alphas(X, y, n_alphas, max_features, binary_response=False):
	n, p = X.shape
	if binary_response:
		y_mean = np.mean(y)
		scaled_residuals = y - y_mean * (1 - y_mean)
		alpha_max = 10 / np.max(np.abs(np.dot(X.T, scaled_residuals) / n))
		selector = LogisticRegression(penalty='l1', solver='liblinear', tol=1e-3, class_weight='balanced')
		if np.isnan(alpha_max):
			alpha_max = 100
		alpha_min = alpha_max * 1e-10
		test_alphas = np.logspace(np.log10(alpha_max/2), np.log10(alpha_min), 100)
	else:
		alpha_max = 2 * np.max(np.abs(np.dot(X.T,y))) / n
		selector = Lasso(warm_start=True)
		if np.isnan(alpha_max):
			alpha_max = 100
		alpha_min = alpha_max * 1e-10
		test_alphas = np.logspace(np.log10(alpha_max/2), np.log10(alpha_min), 100)
	for alpha in test_alphas:
		if binary_response:
			selector.set_params(C=1/alpha)
		else:
			selector.set_params(alpha=alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			selector.fit(X,y)
		num_selected = np.sum(selector.coef_ != 0)
		if num_selected >= max_features:
			alpha_min = alpha
			break
	alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
	return alphas

def compute_correlation(X):
	corr_matrix = np.corrcoef(X, rowvar=False)
	abs_corr_matrix = np.abs(corr_matrix)
	np.fill_diagonal(abs_corr_matrix, 0)
	avg_correlation = np.mean(np.mean(abs_corr_matrix, axis=1))
	max_correlations = np.max(abs_corr_matrix, axis=1)
	avg_max_correlation = np.mean(max_correlations)
	return avg_correlation, avg_max_correlation

def compute_efp_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff):
	n_alphas, p = stability_paths.shape
	if ipss_function not in ['h1', 'h2', 'h3']:
		raise ValueError(f"ipss_function must be 'h1', 'h2', or 'h3', but got ipss_function = {ipss_function} instead")
	m = 1 if ipss_function == 'h1' else 2 if ipss_function == 'h2' else 3
	# function to apply to selection probabilities
	def h_m(x):
		return 0 if x <= 0.5 else (2*x - 1)**m
	# evaluate ipss bounds for specific functions
	if m == 1:
		integral, stop_index = integrate(average_selected**2 / p, alphas, delta, cutoff=cutoff)
	elif m == 2:
		term1 = average_selected**2 / (p * B)
		term2 = (B-1) * average_selected**4 / (B * p**3)
		integral, stop_index  = integrate(term1 + term2, alphas, delta, cutoff=cutoff)
	else:
		term1 = average_selected**2 / (p * B**2)
		term2 = (3 * (B-1) * average_selected**4) / (p**3 * B**2)
		term3 = ((B-1) * (B-2) * average_selected**6) / (p**5 * B**2)
		integral, stop_index = integrate(term1 + term2 + term3, alphas, delta, cutoff=cutoff)
	# compute ipss scores
	alphas_stop = alphas[:stop_index]
	scores = np.zeros(p)
	for i in range(p):
		values = np.empty(stop_index)
		for j in range(stop_index):
			values[j] = h_m(stability_paths[j,i])
		scores[i], _ = integrate(values, alphas_stop, delta)

	return scores, integral, alphas, stop_index

# construct list of average number of features selected cutoffs
def compute_q_list(efp_list, tau, method, p, B, sampling_type='SS'):
	q_list = []
	for efp in efp_list:
		if efp == 0:
			q_list.append(0)
		else:
			q = stabsel_q(p, cutoff=tau, PFER=efp, B=B, assumption=method, sampling_type=sampling_type)
			q_list.append(q)
	return q_list

def compute_qvalues(efp_scores):
	T = list(efp_scores.values())
	fdrs = []
	for t in T:
		efp_scores_leq_t = [score for score in efp_scores.values() if score <= t]
		FP = max(efp_scores_leq_t)
		S = len(efp_scores_leq_t)
		fdrs.append((t, FP/S))
	q_values = {
		feature: min(fdr for t, fdr in fdrs if score <= t)
		for feature, score in efp_scores.items()
	}
	return q_values

def integrate(values, alphas, delta=1, cutoff=None):
	n_alphas = len(alphas)
	a = min(alphas)
	b = max(alphas)
	if delta == 1:
		normalization = (1 - (a/b)**(1/n_alphas)) / np.log(b/a)
	else:
		normalization = (1 - delta) * (1 - (a/b)**(1/n_alphas)) / (b**(1-delta) - a**(1-delta))
	output = 0
	stop_index = n_alphas
	before = stop_index
	if cutoff is None:
		for i in range(1, n_alphas):
			weight = 1 if delta == 1 else alphas[i]**(1-delta)
			output += normalization * weight * values[i-1]
	else:
		for i in range(1, n_alphas):
			weight = 1 if delta == 1 else alphas[i]**(1-delta)
			updated_output = output + normalization * weight * values[i-1]
			if updated_output > cutoff:
				stop_index = i
				break
			else:
				output = updated_output
	return output, stop_index

def score_based_selection(results, n_alphas):
	alpha_min = np.min(results)
	if alpha_min < 0:
		results += np.abs(alpha_min)
	alpha_max = np.max(results) + .01
	alpha_min = alpha_max / 1e8
	alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
	B, _, p = results.shape

	reshape_results = np.empty((B, n_alphas, 2, p))
	for i in range(n_alphas):
		reshape_results[:,i,:,:] = results
	results = reshape_results
	for i, alpha in enumerate(alphas):
		for b in range(B):
			for j in range(2):
				results[b,i,j,:] = (results[b,i,j,:] > alpha).astype(int)
	return results, alphas

# subsampler for estimating selection probabilities
def selection(X, y, alphas, selector, **kwargs):
	n, p = X.shape
	indices = np.arange(n)
	np.random.shuffle(indices)
	n_split = int(len(indices) / 2)

	if alphas is None:
		indicators = np.empty((2,p))
		for half in range(2):
			idx = indices[:n_split] if half == 0 else indices[n_split:]
			indicators[half, :] = np.array(selector(X[idx,:], y[idx], **kwargs))
	else:
		indicators = np.empty((len(alphas), 2, p))
		for half in range(2):
			idx = indices[:n_split] if half == 0 else indices[n_split:]
			indicators[:, half, :] = np.array(selector(X[idx,:], y[idx], alphas, **kwargs))

	return indicators

def selector_and_args(selector, selector_args):
	selectors = {'gb_classifier':fit_gb_classifier, 'gb_regressor':fit_gb_regressor, 'logistic_regression':fit_l1_classifier,
		'lasso':fit_l1_regressor, 'mcp':fit_mcp_regressor, 'rf_classifier':fit_rf_classifier, 'rf_regressor':fit_rf_regressor, 
		'scad':fit_scad_regressor}
	if selector in selectors:
		selector_function = selectors[selector]
		if selector == 'logistic_regression' and not selector_args:
			# selector_args = {'penalty': 'l1', 'solver':'liblinear', 'tol': 1e-3, 'class_weight': 'balanced'}
			selector_args = {'penalty': 'l1', 'solver':'saga', 'tol': 1e-3, 'warm_start': True, 'class_weight': 'balanced'}
		elif selector in ['gb_classifier', 'gb_regressor'] and not selector_args:
			selector_args = {'max_depth':1, 'colsample_bynode':1/3, 'n_estimators':100, 'importance_type':'gain'}
		elif selector in ['rf_classifier', 'rf_regressor'] and not selector_args:
			selector_args = {'max_features':1/10, 'n_estimators':50}
		else:
			selector_args = {}
	else:
		selector_function = selector
	return selector_function, selector_args

def stabsel_q(p, cutoff, PFER, B=50, assumption='unimodal', sampling_type='SS', verbose=False):
	p_r = IntVector([p])
	cutoff_r = FloatVector([cutoff])
	PFER_r = FloatVector([PFER])
	B_r = IntVector([B]) if B is not None else NULL
	assumption_r = StrVector([assumption])
	sampling_type_r = StrVector([sampling_type])
	verbose_r = BoolVector([verbose])
	
	# Call the stabsel_parameters function with PFER and cutoff, letting it find q
	result = stabs.stabsel_parameters(
		p=p_r, cutoff=cutoff_r, PFER=PFER_r, 
		B=B_r, assumption=assumption_r, 
		sampling_type=sampling_type_r, verbose=verbose_r
	)
	
	# Extract the value of q from the result
	q = int(result.rx2('q')[0])
	
	return q

def tps_and_fps(efp_scores, efp_list, true_features):
	tps, fps = [], []
	for efp in efp_list:
		tp, fp = 0, 0
		for j in efp_scores.keys():
			if efp_scores[j] <= efp:
				if j in true_features:
					tp += 1
				else:
					fp += 1
		tps.append(tp)
		fps.append(fp)
	return np.array(tps), np.array(fps)