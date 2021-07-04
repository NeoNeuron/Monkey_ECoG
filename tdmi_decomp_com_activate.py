import numpy as np
import matplotlib.pyplot as plt
import pickle

from fcpy.tdmi import MI_stats, compute_snr_matrix, compute_noise_matrix
from fcpy.utils import Linear_R2

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
# prepare weight_flatten
weight = data_package['weight']
off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)
weight[np.eye(weight.shape[0], dtype=bool)] = 0
# load snr-th
with open(path + 'snr_th.pkl', 'rb') as f:
    snr_th = pickle.load(f)
tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

with open(path + 'WA_v3.pkl', 'rb') as f:
    tdmi_mask_total = pickle.load(f)

seperator = [-6, -5, -4, -3, -2, -1, 0]
title_set = [
    "## $w_{ij}>10^{%d}$ " % item for item in seperator
]
# for title in title_set:
title = title_set[0]
tdmi_sum = np.array([tdmi_mask_total[title][key] for key in filter_pool[:-1]]).sum(0)
snr_mask_2 = tdmi_sum==6

# triu_mask = np.triu(np.ones_like(weight).astype(bool), k=1)

mi_data = np.zeros((len(filter_pool), np.sum(snr_mask_2).astype(int)))

for i, band in enumerate(filter_pool):
    # mi_data[i,:] = tdmi_data[band][triu_mask,1000]
    tdmi_data_band = MI_stats(tdmi_data[band], 'max')
    noise_matrix = compute_noise_matrix(tdmi_data[band])
    snr_matrix = compute_snr_matrix(tdmi_data[band])
    snr_mask = snr_matrix > snr_th[band]
    tdmi_data_band[~snr_mask] = noise_matrix[~snr_mask]
    mi_data[i,:] = tdmi_data_band[snr_mask_2]

fig, ax = plt.subplots(7,1, figsize=(3,20), sharex=True)
for i in range(6):
    ax[i].loglog(mi_data[i,:], mi_data[-1,:], '.k', ms=.4)
    pval = np.polyfit(mi_data[i,:], mi_data[-1,:], deg=1)
    r2 = Linear_R2(mi_data[i,:], mi_data[-1,:], pval)
    ax[i].loglog([mi_data[i,:].min(),mi_data[i,:].max()], np.polyval(pval, [mi_data[i,:].min(), mi_data[i,:].max()]), '--', color='orange')
    ax[i].set_xlabel(r'$MI_{band}$')
    ax[i].set_ylabel(r'$MI_{raw}$')
    ax[i].set_title(f'{filter_pool[i]:s}, r={np.sqrt(r2):.3f}')

# least square methods
x,res,rank,_ = np.linalg.lstsq(mi_data[:-1,:].T, mi_data[-1,:])
print(x)
print(res)
print(rank)
lsq_lhs = mi_data[:-1,:].T @ x
ax[-1].loglog(lsq_lhs, mi_data[-1,:], '.k', ms=.4)
ax[-1].set_xlabel(r'$\sum MI_{band}$')
ax[-1].set_ylabel(r'$MI_{raw}$')
r2 = Linear_R2(lsq_lhs, mi_data[-1,:], [1,0])
ax[-1].loglog([lsq_lhs.min(),lsq_lhs.max()], [lsq_lhs.min(), lsq_lhs.max()], '--', color='orange')
ax[-1].set_title(f'Least Square Fit: r={np.sqrt(r2):.3f}')
plt.tight_layout()
plt.savefig(path + 'tdmi_decomp_com_activate.png')