#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from utils.tdmi import *
import pickle

path = 'tdmi_snr_analysis/'
data_package = np.load(path+'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
# prepare weight_flatten
weight = data_package['weight']
with open(path+'snr_th.pkl', 'rb') as f:
    snr_th = pickle.load(f)

tdmi_data = np.load(path + 'tdmi_data_long.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

fig, ax = plt.subplots(2, len(filter_pool), figsize=(30,8))
for i, band in enumerate(filter_pool):
    tdmi_full = compute_tdmi_full(tdmi_data[band])
    # noise_matrix = compute_noise_matrix(tdmi_data[band])
    snr_matrix = compute_snr_matrix(tdmi_data[band])
    snr_mask = snr_matrix > snr_th[band]

    delay = np.arange(tdmi_full.shape[2])-(tdmi_full.shape[2]-1)/2
    n = 5
    idx = np.random.choice(np.arange((~snr_mask).sum()), n, replace=False)
    [ax[0, i].plot(delay, trace, color='k', alpha=.5) for trace in tdmi_full[~snr_mask][idx]]
    snr_mask *= snr_matrix <= snr_th[band]*1.5 
    idx = np.random.choice(np.arange(snr_mask.sum()), n, replace=False)
    [ax[0, i].plot(delay, trace, color='r', alpha=.3) for trace in tdmi_full[snr_mask][idx]]
    # ax[0,i].plot(delay, tdmi_full[~snr_mask].mean(0), color='k', alpha=1)
    # ax[0,i].plot(delay, tdmi_full[ snr_mask].mean(0), color='r', alpha=1)
    ax[0, i].set_xlabel('Time delay (ms)')
    ax[0, i].set_ylabel('Mutual Info (nats)')
    ax[0, i].set_xlim(-500,500)
    ax[0, i].set_title(band)

    snr_matrix[np.eye(117, dtype=bool)] = np.nan
    ax[1, i].hist(np.log10(snr_matrix.flatten()), bins=100)
    ax[1, i].axvline(np.log10(snr_th[band]), color = 'orange')
    ax[1, i].set_xlabel(r'$\log_{10}$(SNR)')
    ax[1, i].set_ylabel('Counts')
plt.tight_layout()
fig.savefig(path + 'snr_distribution_and_tdmi_trace.png')
