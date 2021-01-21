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
tdmi_mode = 'sum' # or max
# tdmi_mode = 'max' # or sum

def Linear_R2(x, y, pval):
    y_predict = x*pval[0]+pval[1]
    R = np.corrcoef(y, y_predict)[0,1]
    return R**2

fig, ax = plt.subplots(2,3,figsize=(20,10))
ax = ax.reshape((6,))
for idx, band in enumerate(filter_pool):
    tdmi_data = np.load('sum_tdmi_processing_1217/data/data_r2_'+band+'_tdmi_126-10_total.npy')
    if tdmi_mode == 'sum':
        # sum tdmi mode
        tdmi_data = tdmi_data[:,:,:10].sum(2)
    elif tdmi_mode == 'max':
        # max tdmi mode
        tdmi_data = tdmi_data.max(2)
        # bin counts = [673,2,53,120,81,123,110,51,47]
    else:
        raise ValueError('Invalid tdmi_mode')
    log_tdmi_data = np.log10(tdmi_data)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    answer = data_package['con_known']
    # pval, cov = np.polyfit(answer.flatten(), log_tdmi_data.flatten(), deg=1,cov=True)
    # answer_set = np.unique(answer.flatten())
    # log_tdmi_data_mean = np.array([np.mean(log_tdmi_data.flatten()[answer.flatten()==key]) for key in answer_set])
    answer[answer==0]=1e-7
    log_answer = np.log10(answer)
    answer_edges = np.linspace(-7, 1, num = 10)
    log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
    for i in range(len(answer_edges)-1):
        mask = (log_answer.flatten() >= answer_edges[i]) & (log_answer.flatten() < answer_edges[i+1])
        print(mask.sum())
        log_tdmi_data_mean[i] = np.mean(log_tdmi_data.flatten()[mask])
    pval, cov = np.polyfit(answer_edges[:-1], log_tdmi_data_mean, deg=1,cov=True)
    # ax[idx].plot(np.log10(answer.flatten()), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[idx].plot(answer_edges[:-1], log_tdmi_data_mean, 'k.', markersize=15, label='TDMI mean')
    ax[idx].plot(answer_edges[:-1], np.polyval(pval, answer_edges[:-1]), 'r', label='Linear Fitting')
    if tdmi_mode == 'sum':
        ax[idx].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
    elif tdmi_mode == 'max':
        ax[idx].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
    ax[idx].set_xlabel(r'$log_{10}$(Connectivity Strength)')
    ax[idx].set_title(f'{band:s} $R^2$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval):5.3f}')
    ax[idx].legend(fontsize=15)
    ax[idx].grid(ls='--')

plt.tight_layout()
plt.savefig(f'data_r2_mi-s_{tdmi_mode:s}.png')