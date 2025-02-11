# Baseline feature selection algorithms for IPSS and stability selection
"""
Baseline selectors based on:
	- l1-regularized linear (lasso) or logistic regression
	- minimax concave penalty (MCP)
	- smoothly clipped absolute deviation penalty (SCAD)
"""

import warnings

import numpy as np
from skglm.estimators import GeneralizedLinearEstimator, MCPRegression
from skglm.penalties import SCAD
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression

# l1-regularized logistic regression
def fit_l1_classifier(X, y, alphas, **kwargs):
	model = LogisticRegression(**kwargs)
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		model.set_params(C=1/alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			model.fit(X, y)
			coefficients[i,:] = (model.coef_ != 0).astype(int)
	return coefficients

# l1-regularized linear regression (lasso)
def fit_l1_regressor(X, y, alphas, **kwargs):
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		_, coefs, _ = lasso_path(X, y, alphas=alphas, **kwargs)
	return (coefs.T != 0).astype(int)

# minimax concave penalty (MCP)
def fit_mcp_regressor(X, y, alphas, gamma=3.0):
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		model = MCPRegression(alpha=alpha, gamma=gamma)
		model.fit(X, y)
		coefficients[i, :] = (model.coef_ != 0).astype(int)
	return coefficients

# smoothly clipped absolute deviation penalty (SCAD)
def fit_scad_regressor(X, y, alphas, gamma=3.7):
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		penalty = SCAD(alpha=alpha, gamma=gamma)
		model = GeneralizedLinearEstimator(penalty=penalty)
		model.fit(X, y)
		coefficients[i, :] = (model.coef_ != 0).astype(int)
	return coefficients


