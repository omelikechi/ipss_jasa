# Analyze colon cancer data
"""
The results are reported in Section 6.2. Here, we provide code for selecting the genes related
to colon cancer (cancerous versus normal tissue) and for plotting the associated stability paths
and gene expression heatmap. Note that some of the genes selected by each method can change with
the random seed; however, many of the genes selected by the different methods are consistent
across different random seeds.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from helpers import compute_q_list, tps_and_fps
from main import lassocv, mcpcv, scadcv, select

# set random seed
random_seed = 25
np.random.seed(random_seed)

#--------------------------------
# Load data
#--------------------------------
data = np.load('./applications/colon_data.npy', allow_pickle=True).item()
X, y, feature_names = data['X'], data['y'], data['feature_names']
sorted_indices = np.argsort(y)
y = y[sorted_indices] 
X = X[sorted_indices] 
n, p = X.shape

#--------------------------------
# Methods specifications
#--------------------------------
target_efp = 1/2
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

#--------------------------------
# Run analysis
#--------------------------------
results = select(X, y, ipss_args, ss_args, **select_args)
ipss_scores = results['ipss_efp_scores_list']
ss_scores = results['ss_efp_scores_list']

#--------------------------------
# Results
#--------------------------------
selected_genes = {}

for function in ipss_args['function_list']:
	for cutoff in ipss_args['cutoff_list']:
		for delta in ipss_args['delta_list']:
			scores = ipss_scores[function][cutoff][delta]
			total_selected = 0
			print(f'Genes selected by IPSS({function}):')
			for j in scores.keys():
				if scores[j] <= target_efp:
					total_selected += 1
					selected_genes[j] = feature_names[j]
					print(f'  {feature_names[j]}')
			print(f'Total number selected: {total_selected}')
	print()

for assumption in ss_args['assumption_list']:
	for tau in ss_args['tau_list']:
		scores = ss_scores[assumption][tau]
		total_selected = 0
		print(f'Genes selected by SS({assumption_name[assumption]}):')
		for j in scores.keys():
				if scores[j] <= target_efp:
					total_selected += 1
					selected_genes[j] = feature_names[j]
					print(f'  {feature_names[j]}')
		print(f'Total number selected: {total_selected}')
	print()

#--------------------------------
# Plot stability paths
#--------------------------------
stability_paths = results['stability_paths']

n_alphas, p = stability_paths.shape

# cut stability paths short for better visualization
n_alphas = int(n_alphas / 2)

def generate_rainbow_colors(n):
	colors = plt.cm.hsv(np.linspace(0, 1, n))
	np.random.shuffle(colors)
	return colors

colors = generate_rainbow_colors(len(selected_genes))
lws = [2 if j in selected_genes else 1 for j in range(p)]

color_idx = 0
for j in range(p):
	if j in selected_genes.keys():
		plt.plot(np.arange(n_alphas), stability_paths[:n_alphas,j], color=colors[color_idx], label=selected_genes[j], lw=lws[j])
		color_idx += 1
	else:
		plt.plot(np.arange(n_alphas), stability_paths[:n_alphas,j], color='gray', lw=lws[j])

plt.xticks([])
plt.xlabel(f'$-\\log(\\lambda)$', fontsize=18)

plt.legend(loc='best')
plt.tight_layout()
plt.show()

#--------------------------------
# Plot heatmap of selected genes
#--------------------------------
indices = list(selected_genes.keys())  
names = list(selected_genes.values())  

# Subset the data to only include selected features
X_selected = X[:, indices]

# Define sample group sizes
num_cancerous = 40
num_normal = 22

# Calculate average expression for cancerous samples
avg_expression_cancerous = np.mean(X_selected[:num_cancerous, :], axis=0)

# Sort indices based on average expression in cancerous samples
sorted_indices = np.argsort(avg_expression_cancerous)
X_sorted = X_selected[:, sorted_indices]
sorted_names = [names[i] for i in sorted_indices]

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X_sorted, cmap='binary', cbar=True)

# Draw a horizontal line to separate cancerous and normal samples
plt.axhline(y=num_cancerous, color='red', linestyle='--', linewidth=3)

# Annotate the regions with labels
plt.text(-0.5, num_cancerous / 2, 'Cancerous', ha='center', va='center', rotation='vertical', fontsize=18)
plt.text(-0.5, num_cancerous + num_normal / 2, 'Normal', ha='center', va='center', rotation='vertical', fontsize=18)

# Set tick rotation for x-axis and apply bold labels
plt.xticks(np.arange(len(sorted_names)), sorted_names, rotation=45, ha='left')
for tick in plt.gca().get_xticklabels():
	tick.set_color('black')
	tick.set_weight('bold')

# Remove y-axis tick labels
plt.yticks([])

plt.title('Expression Levels of Selected Genes', fontsize=18)
plt.tight_layout()
plt.show()











