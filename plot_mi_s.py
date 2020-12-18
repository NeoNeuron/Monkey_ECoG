#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
import matplotlib.pyplot as plt

data_package = np.load('preprocessed_data.npz')

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
fig, ax = plt.subplots(2,3,figsize=(20,10), sharex=True)
ax = ax.reshape((6,))
for idx, band in enumerate(filter_pool):
    tdmi_data = np.load('sum_tdmi_processing_1217/data/data_r2_'+band+'_tdmi_126-10_total.npy')
    print(tdmi_data.shape)
    # sum tdmi mode
    tdmi_data = tdmi_data[:,:,:10].sum(2)
    # max tdmi mode
    # tdmi_data = tdmi_data.max(2)
    log_tdmi_data = np.log10(tdmi_data)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    answer = data_package['con_known']
    # pval, cov = np.polyfit(answer.flatten(), log_tdmi_data.flatten(), deg=1,cov=True)
    # answer_set = np.unique(answer.flatten())
    # log_tdmi_data_mean = np.array([np.mean(log_tdmi_data.flatten()[answer.flatten()==key]) for key in answer_set])
    answer[answer==0]=1e-6
    log_answer = np.log10(answer)
    answer_edges = np.linspace(-6, 1, num = 9)
    log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
    for i in range(len(answer_edges)-1):
        mask = (log_answer.flatten() >= answer_edges[i]) & (log_answer.flatten() < answer_edges[i+1])
        log_tdmi_data_mean[i] = np.mean(log_tdmi_data.flatten()[mask])
    pval, cov = np.polyfit(answer_edges[:-1], log_tdmi_data_mean, deg=1,cov=True)
    # ax[idx].plot(np.log10(answer.flatten()), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[idx].plot(answer_edges[:-1], log_tdmi_data_mean, 'k.', markersize=15, label='TDMI mean')
    ax[idx].plot(answer_edges[:-1], np.polyval(pval, answer_edges[:-1]), 'r', label='Linear Fitting')
    ax[idx].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
    ax[idx].set_xlabel(r'$log_{10}$(Connectivity Strength)')
    ax[idx].set_title(f'{band:s} $R^2$ = {cov[0,0]:5.3e}')
    ax[idx].legend(fontsize=15)
    ax[idx].grid(ls='--')

plt.tight_layout()
plt.savefig('data_r2_mi-s.png')