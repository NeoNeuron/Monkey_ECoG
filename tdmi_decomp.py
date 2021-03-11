import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils.tdmi import MI_stats, compute_snr_matrix, compute_noise_matrix

path = 'tdmi_snr_analysis/'
data_package = np.load(path+'preprocessed_data.npz', allow_pickle=True)
# prepare weight_flatten
weight = data_package['weight']
off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)
weight[np.eye(weight.shape[0], dtype=bool)] = 0
# load snr-th
with open(path + 'snr_th.pkl', 'rb') as f:
    snr_th = pickle.load(f)
tdmi_data = np.load(path + 'tdmi_data_long.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

triu_mask = np.triu(np.ones_like(weight).astype(bool), k=1)

mi_data = np.zeros((len(filter_pool), np.sum(triu_mask).astype(int)))

for i, band in enumerate(filter_pool):
    # mi_data[i,:] = tdmi_data[band][triu_mask,1000]
    tdmi_data_band = MI_stats(tdmi_data[band], 'max')
    noise_matrix = compute_noise_matrix(tdmi_data[band])
    snr_matrix = compute_snr_matrix(tdmi_data[band])
    snr_mask = snr_matrix > snr_th[band]
    tdmi_data_band[~snr_mask] = noise_matrix[~snr_mask]
    mi_data[i,:] = tdmi_data_band[triu_mask]

fig, ax = plt.subplots(1,7, figsize=(25,3))
for i in range(6):
    ax[i].loglog(mi_data[i,:], mi_data[-1,:], '.k', ms=.4)
    ax[i].set_xlabel(r'$MI_{band}$')
    ax[i].set_ylabel(r'$MI_{raw}$')
    ax[i].set_title(filter_pool[i])

# least square methods
x,res,rank,_ = np.linalg.lstsq(mi_data[:-1,:].T, mi_data[-1,:])
print(x)
print(res)
print(rank)
ax[-1].loglog(mi_data[:-1,:].T @ x, mi_data[-1,:], '.k', ms=.4)
ax[-1].set_xlabel(r'$\sum MI_{band}$')
ax[-1].set_ylabel(r'$MI_{raw}$')
ax[-1].set_title('Least Square Fit')
plt.tight_layout()
plt.savefig(path + 'tdmi_decomp.png')