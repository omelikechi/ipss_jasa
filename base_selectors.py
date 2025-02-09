# Baseline feature selection algorithms for IPSS
"""
Baseline selectors based on:
	- Importance scores from gradient boosting with XGBoost
	- l1-regularized linear (lasso) or logistic regression
	- Importance scores from random forests with sci-kit learn
"""

import warnings

import numpy as np
from skglm.estimators import GeneralizedLinearEstimator, MCPRegression
from skglm.penalties import SCAD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

# gradient boosting classifier
def fit_gb_classifier(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gain')
	seed = np.random.randint(1e5)
	model = xgb.XGBClassifier(random_state=seed, **kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances

# gradient boosting regressor
def fit_gb_regressor(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gain')
	seed = np.random.randint(1e5)
	model = xgb.XGBRegressor(random_state=seed, **kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances

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

def fit_mcp_regressor(X, y, alphas, gamma=3.0):
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		model = MCPRegression(alpha=alpha, gamma=gamma)
		model.fit(X, y)
		coefficients[i, :] = (model.coef_ != 0).astype(int)
	return coefficients

# random forest classifier
def fit_rf_classifier(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gini')
	model = RandomForestClassifier(class_weight='balanced', **kwargs)
	model.fit(X, y)
	if importance_type == 'gini' or importance_type is None:
		feature_importances = model.feature_importances_
	elif importance_type == 'permutation':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
		perm_importance = permutation_importance(model, X_test, y_test, n_repeats=1)
		feature_importances = perm_importance.importances_mean
	else:
		raise ValueError("importance_type must be either 'gini' or 'permutation'")
	return feature_importances

# random forest regressor
def fit_rf_regressor(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gini')
	model = RandomForestRegressor(**kwargs)
	model.fit(X, y)
	if importance_type == 'gini':
		feature_importances = model.feature_importances_
	elif importance_type == 'permutation':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		perm_importance = permutation_importance(model, X_test, y_test, n_repeats=1)
		feature_importances = perm_importance.importances_mean
	elif importance_type == 'shadow':
		n, p = X.shape
		X_shadow = X.copy()
		for i in range(p):
			np.random.shuffle(X_shadow[:,i])
		X_combined = np.hstack((X, X_shadow))
		model.fit(X_combined, y)
		importances_combined = model.feature_importances_
		n_features = X.shape[1]
		original_importances = importances_combined[:n_features]
		shadow_importances = importances_combined[n_features:]
		feature_importances = original_importances - shadow_importances
	else:
		raise ValueError("importance_type must be either 'gini', 'permutation', or 'shadow'")
	return feature_importances

# scad regressor
def fit_scad_regressor(X, y, alphas, gamma=3.7):
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		penalty = SCAD(alpha=alpha, gamma=gamma)
		model = GeneralizedLinearEstimator(penalty=penalty)
		model.fit(X, y)
		coefficients[i, :] = (model.coef_ != 0).astype(int)
	return coefficients


