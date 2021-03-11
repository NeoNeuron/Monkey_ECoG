
# ! Author: Kai Chen
# Plot the distribution of number of reconstructed edges over different band.
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.1

path = 'tdmi_snr_analysis/'
data_package = np.load(path+'preprocessed_data.npz', allow_pickle=True)
weight = data_package['adj_mat']
off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
seperator = [-6, -5, -4, -3, -2, -1, 0]

with open(path + 'WA_v3_cg.pkl', 'rb') as f:
    tdmi_mask_total = pickle.load(f)
title_set = [
    "## $w_{ij}>10^{%d}$ " % item for item in seperator
]
fig, ax = plt.subplots(1, len(seperator), figsize=(26, 3))
for idx, title in enumerate(title_set):
    # plot figure
    tdmi_sum = np.array([tdmi_mask_total[title][key] for key in filter_pool[:-1]]).sum(0)
    tdmi_raw = tdmi_mask_total[title]['raw'].copy()
    raw_number = np.zeros(6)
    for n in np.arange(6)+1:
        raw_number[n-1] = tdmi_raw[tdmi_sum==n].sum()

    counts, edges = np.histogram(tdmi_sum[off_diag_mask].flatten(), bins=7)
    ax[idx].bar(np.arange(counts.shape[0]-1)+1, height=counts[1:], tick_label=np.arange(counts.shape[0]-1)+1)
    ax[idx].bar(np.arange(counts.shape[0]-1)+1, height=raw_number, color=(1,1,1,0), edgecolor='r', ls='--', label='Matched with RAW')
    ax[idx].set_xticklabels(np.arange(counts.shape[0]-1)+1)
    ax[idx].set_title(title.strip('#'))
ax[0].legend()

plt.tight_layout()
plt.savefig(path + f'WA_v4_cg_1.png')
plt.close()