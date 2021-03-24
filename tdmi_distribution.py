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
off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)
weight[np.eye(weight.shape[0], dtype=bool)] = 0
# load snr-th
with open(path + 'snr_th.pkl', 'rb') as f:
    snr_th = pickle.load(f)
tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# load various of thresholds
pval = np.load(path + 'pval.npz', allow_pickle=True)
with open(path + 'opt_threshold_channel_tdmi_max.pkl', 'rb') as f:
    roc_th = pickle.load(f)
with open(path + 'gap_th.pkl', 'rb') as f:
    gap_th = pickle.load(f)

fig, ax = plt.subplots(len(filter_pool), 5, figsize=(20,20), sharex='row')
for i, band in enumerate(filter_pool):
    noise_matrix = compute_noise_matrix(tdmi_data[band])
    snr_matrix = compute_snr_matrix(tdmi_data[band])
    snr_mask = snr_matrix > snr_th[band]
    tdmi_data_band = MI_stats(tdmi_data[band], 'max')
    tdmi_data_band[~snr_mask] = noise_matrix[~snr_mask]

    snr_matrix[np.eye(117, dtype=bool)] = np.nan
    ax[i,0].hist(np.log10(tdmi_data_band[off_diag_mask]), bins=100)
    ax[i,0].axvline(roc_th[band][0], color = 'orange', label='roc_th')
    ax[i,0].axvline(gap_th[band], color = 'r', label='gap_th')
    pval_th = pval[band][0]*1e-6 + pval[band][1]
    ax[i,0].axvline(pval_th, color = 'g', label='pval_th')
    ax[i,0].set_title(band)
    ax[-1,0].set_xlabel(r'$\log_{10}$(SNR)')
    ax[-1,0].set_ylabel('Counts')

    threshold_options = [2, 1e-2, 1e-3, 1e-4, 1e-5]
    
    for j in range(len(threshold_options)-1):
        weight_mask = (weight < threshold_options[j])*(weight>=threshold_options[j+1])
        ax[i, j+1].hist(np.log10(tdmi_data_band[weight_mask*off_diag_mask]), bins=100)
        ax[i, j+1].axvline(roc_th[band][-3-j], color = 'orange', label='roc_th')
        ax[i, j+1].axvline(gap_th[band], color = 'r', label='gap_th')
        pval_th = pval[band][0]*threshold_options[j+1] + pval[band][1]
        ax[i, j+1].axvline(pval_th, color = 'g', label='pval_th')
        ax[i, j+1].set_ylim(0,80)
        ax[i, j+1].set_yticks([0,20,40,60,80])
        ax[0, j+1].set_title(f'{threshold_options[j+1]:.0e} <= w < {threshold_options[j]:.0e}')
        ax[i, j+1].grid(ls='--')
        
plt.tight_layout()
fig.savefig(path + 'tdmi_distribution.eps')

