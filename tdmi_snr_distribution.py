#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from utils.tdmi import *
import pickle

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
# prepare weight_flatten
weight = data_package['weight']
weight[np.eye(weight.shape[0], dtype=bool)] = 0
off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)
snr_th = {
    'delta'      :1.5,
    'theta'      :2.0,
    'alpha'      :2.0,
    'beta'       :3.0,
    'gamma'      :5,  
    'high_gamma' :5,  
    'raw'        :4.0,
}
tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

fig, ax = plt.subplots(len(filter_pool), 5, figsize=(20,20), sharex='row')
for i, band in enumerate(filter_pool):
    # noise_matrix = compute_noise_matrix(tdmi_data[band])
    snr_matrix = compute_snr_matrix(tdmi_data[band])

    ax[i,0].hist(np.log10(snr_matrix[off_diag_mask]), bins=100)
    ax[i,0].axvline(np.log10(snr_th[band]), color = 'orange')
    ax[i,0].set_title(band)
    ax[-1,0].set_xlabel(r'$\log_{10}$(SNR)')
    ax[-1,0].set_ylabel('Counts')

    threshold_options = [2, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for j in range(len(threshold_options)-1):
        weight_mask = (weight < threshold_options[j])*(weight>=threshold_options[j+1])
        ax[i, j+1].hist(np.log10(snr_matrix[off_diag_mask*weight_mask]), bins=60)
        ax[i, j+1].axvline(np.log10(snr_th[band]), color = 'orange')
        ax[i, j+1].set_ylim(0,80)
        ax[i, j+1].set_yticks([0,20,40,60,80])
        ax[i, j+1].grid(ls='--')
        ax[i, j+1].text(
            0.95, 0.95, 
            f'Above ratio:\n{(snr_matrix[off_diag_mask*weight_mask]>snr_th[band]).sum()*100./weight_mask[off_diag_mask].sum():4.1f} %',
            fontsize=14,
            transform=ax[i, j+1].transAxes, 
            verticalalignment='top', horizontalalignment='right'
        )

    [ax[0, j+1].set_title(f'{threshold_options[j+1]:.0e} < w <= {threshold_options[j]:.0e}') for j in range(len(threshold_options)-1)]
        
plt.tight_layout()
fig.savefig(path + 'snr_distribution.eps')

# save snr-th
with open(path + 'snr_th.pkl', 'wb') as f:
    pickle.dump(snr_th, f)
