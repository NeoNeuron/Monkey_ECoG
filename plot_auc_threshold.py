#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
import matplotlib.pyplot as plt

data_package = np.load('preprocessed_data.npz')

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
fig, ax = plt.subplots(2,3,figsize=(20,10), sharey=True)
ax = ax.reshape((6,))
threshold_options = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for idx, band in enumerate(filter_pool):
    tdmi_data = np.load('sum_tdmi_processing_1217/data/data_r2_'+band+'_tdmi_126-10_total.npy')
    print(tdmi_data.shape)
    # sum tdmi mode
    tdmi_data = tdmi_data[:,:,:10].sum(2)
    # max tdmi mode
    # tdmi_data = tdmi_data.max(2)
    log_tdmi_data = np.log10(tdmi_data)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    aucs = np.zeros_like(threshold_options)
    for iidx, threshold in enumerate(threshold_options):
        answer = data_package['con_known']
        answer = (answer>threshold).astype(bool)
        false_positive = np.array([np.sum((log_tdmi_data>i)*(~answer))/np.sum(~answer) for i in np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)])
        true_positive = np.array([np.sum((log_tdmi_data>i)*(answer))/np.sum(answer) for i in np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)])
        auc = -np.sum(np.diff(false_positive)*(true_positive[:-1]+true_positive[1:])/2)
        aucs[iidx] = auc
    ax[idx].semilogx(threshold_options, aucs, '-*')
    ax[idx].set_xlabel('Threshold value')
    ax[idx].set_ylabel('AUC')
    ax[idx].set_title(band)
    ax[idx].grid(ls='--')

plt.tight_layout()
plt.savefig('data_r2_auc-threshold.png')