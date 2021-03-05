#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from utils.tdmi import *
import pickle

path = 'tdmi_snr_analysis/'
data_package = np.load(path+'preprocessed_data.npz', allow_pickle=True)
# prepare weight_flatten
weight = data_package['weight']
weight[np.eye(weight.shape[0], dtype=bool)] = 0
snr_th = {
    'delta'      :3.5,
    'theta'      :5.0,
    'alpha'      :5.0,
    'beta'       :6.5,
    'gamma'      :20,
    'high_gamma' :20,
    'raw'        :8.0,
}
tdmi_data = np.load(path + 'tdmi_data_long.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

fig, ax = plt.subplots(len(filter_pool), 5, figsize=(20,20), sharex='row')
for i, band in enumerate(filter_pool):
    tdmi_full = compute_tdmi_full(tdmi_data[band])
    # noise_matrix = compute_noise_matrix(tdmi_data[band])
    snr_matrix = compute_snr_matrix(tdmi_data[band])

    snr_matrix[np.eye(117, dtype=bool)] = np.nan
    ax[i,0].hist(np.log10(snr_matrix.flatten()), bins=100)
    ax[i,0].axvline(np.log10(snr_th[band]), color = 'orange')
    ax[i,0].set_title(band)
    ax[-1,0].set_xlabel(r'$\log_{10}$(SNR)')
    ax[-1,0].set_ylabel('Counts')

    threshold_options = [2, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for j in range(len(threshold_options)-1):
        weight_mask = (weight < threshold_options[j])*(weight>=threshold_options[j+1])
        snr_matrix_buffer = snr_matrix.copy()
        snr_matrix_buffer[~weight_mask] = np.nan
        ax[i, j+1].hist(np.log10(snr_matrix_buffer.flatten()), bins=60)
        ax[i, j+1].axvline(np.log10(snr_th[band]), color = 'orange')
        ax[i, j+1].set_ylim(0,80)
        ax[i, j+1].set_yticks([0,20,40,60,80])
        ax[0, j+1].set_title(f'{threshold_options[j]:.0e} < w <= {threshold_options[j+1]:.0e}')
        ax[i, j+1].grid(ls='--')
        
plt.tight_layout()
fig.savefig(path + 'snr_distribution.eps')

# save snr-th
with open(path + 'snr_th.pkl', 'wb') as f:
    pickle.dump(snr_th, f)
