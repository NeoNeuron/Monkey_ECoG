
# ! Author: Kai Chen
# Plot the distribution of number of reconstructed edges over different band.
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.1

path = 'tdmi_snr_analysis/'
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
seperator = [-6, -5, -4, -3, -2, -1, 0]

with open(path + 'recon_fit_tdmi_CG.pkl', 'rb') as f:
    _ = pickle.load(f)
    tdmi_mask_total = pickle.load(f)
fig, ax = plt.subplots(1, len(seperator), figsize=(26, 3))
for idx in range(len(seperator)):
    tdmi_sum = np.array([tdmi_mask_total[key][idx] for key in filter_pool[:-1]]).sum(0)
    tdmi_raw = tdmi_mask_total['raw'][idx].copy()
    raw_number = np.zeros(6)
    for n in np.arange(6)+1:
        raw_number[n-1] = tdmi_raw[tdmi_sum==n].sum()

    counts, edges = np.histogram(tdmi_sum, bins=7)
    ax[idx].bar(np.arange(counts.shape[0]-1)+1, height=counts[1:], tick_label=np.arange(counts.shape[0]-1)+1)
    ax[idx].bar(np.arange(counts.shape[0]-1)+1, height=raw_number, color=(1,1,1,0), edgecolor='r', ls='--', label='Matched with RAW')
    ax[idx].set_xticklabels(np.arange(counts.shape[0]-1)+1)
    ax[idx].set_title("$w_{ij}>10^{%d}$ " % seperator[idx])
ax[0].legend()

plt.tight_layout()
plt.savefig(path + f'WA_v4_cg_1.png')
plt.close()