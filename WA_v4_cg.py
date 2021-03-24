
# ! Author: Kai Chen
# Plot the distribution of number of reconstructed edges over different band.
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.1

path = 'tdmi_snr_analysis/'
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
seperator = [-6, -5, -4, -3, -2, -1, 0]

with open(path + 'WA_v3_cg.pkl', 'rb') as f:
    tdmi_mask_total = pickle.load(f)
title_set = [
    "## $w_{ij}>10^{%d}$ " % item for item in seperator
]
fig, ax = plt.subplots(1, len(seperator), figsize=(26, 3))
for idx, title in enumerate(title_set):
    # plot figure
    tdmi_sum = np.array([tdmi_mask_total[title][key] for key in filter_pool[:-1]])
    ax[idx].bar(range(6), height=tdmi_sum.sum(1).sum(1), tick_label=filter_pool[:-1])
    ax[idx].set_xticklabels(filter_pool[:-1], rotation=45)
    ax[idx].set_title(title.strip('#'))

plt.tight_layout()
plt.savefig(path + f'WA_v4_cg.png')
plt.close()