# Generate data for 

import numpy as np
from sklearn.preprocessing import StandardScaler

#--------------------------------
# Main data generator
#--------------------------------
def generate_data(n, p, n_true, snr, response_type, feature_matrix, df=False, function_list=None, 
	standardize=True, center_response=True, random_seed=None):

	np.random.seed(random_seed)

	# randomly select parameter from list
	if isinstance(p, list):
		p = np.random.randint(p[0], p[1] + 1)
	if isinstance(n, list):
		n = np.random.randint(n[0], n[1])
	if isinstance(n_true, list):
		n_true = np.random.randint(n_true[0], n_true[1] + 1)
	elif n_true is None:
		n_true = sample_n_true(p)
	if isinstance(snr, list):
		snr = np.random.uniform(snr[0], snr[1])

	X = generate_features(n, p, feature_matrix, standardize)

	true_features = np.random.choice(np.arange(p), size=n_true, replace=False)

	y = generate_response(X, true_features, snr, response_type, function_list, center_response, df)

	return X, y, true_features

#--------------------------------
# Helpers
#--------------------------------
def generate_features(n, p, feature_matrix, standardize):

	if isinstance(feature_matrix, np.ndarray):
		n_full, p_full = feature_matrix.shape
		if n < n_full:
			rows = np.random.choice(n_full, size=n, replace=False)
			X = feature_matrix[rows, :]
		if p < p_full:
			cols = np.random.choice(p_full, size=p, replace=False)
			X = X[:, cols]
	elif feature_matrix == 0:
		X = np.random.normal(0, 1, size=(n, p))
	elif isinstance(feature_matrix, (int, float)) and 0 <= feature_matrix <= 1:
		indices = np.arange(p)
		Sigma = feature_matrix ** np.abs(indices[:, None] - indices)
		X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
	else:
		raise ValueError("Invalid feature_matrix parameter. Must be a valid NumPy array or a number between 0 and 1.")

	# Standardize features if requested
	if standardize:
		X = StandardScaler().fit_transform(X)

	return X

def generate_response(X, true_features, snr, response_type, function_list, center_response, df):

	n, p = X.shape
	n_true = len(true_features)

	# signal
	beta = np.zeros(p)
	# beta[true_features] = np.random.normal(0, 1, size=n_true)
	beta[true_features] = np.random.uniform(1/2, 1, size=n_true) * np.random.choice([-1, 1], size=n_true)
	signal = X @ beta

	# add noise
	if 'reg' in response_type:
		sigma2 = np.var(signal) / snr
		if df:
			y = signal + np.random.standard_t(df, size=n) * np.sqrt(sigma2)
		else:
			y = signal + np.random.normal(0, np.sqrt(sigma2), size=n)
		y = y - np.mean(y) if center_response else y
	elif 'class' in response_type:
		prob = 1 / (1 + np.exp(-snr * signal))
		y = np.random.binomial(1, prob, size=n)

	return y

def sample_n_true(p, prob=0.9):
	if np.random.rand() < prob:
		sample = np.random.randint(2, 16)
	else:
		sample = np.random.randint(min(15, p//5), min(p//5,50) + 1)

	return sample

def generate_fourier_function(X, n_terms=100, max_frequency=2):
	a_coeffs = np.random.normal(0, 1, n_terms)
	b_coeffs = np.random.normal(0, 1, n_terms)
	frequencies = np.random.randint(1, max_frequency + 1, n_terms)
	y = np.zeros_like(X)
	for a, b, freq in zip(a_coeffs, b_coeffs, frequencies):
		y += a * np.cos(freq * X) + b * np.sin(freq * X)
	y = (y - np.mean(y)) / np.std(y)
	return y



