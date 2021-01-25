#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=15
mpl.rcParams['axes.labelsize'] = 15
import matplotlib.pyplot as plt

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

tdmi_mode = 'sum' # or max

def Linear_R2(x, y, pval):
    mask = ~np.isnan(y)
    y_predict = x[mask]*pval[0]+pval[1]
    R = np.corrcoef(y[mask], y_predict)[0,1]
    return R**2

adj_mat = data_package['adj_mat']
weight_flatten = adj_mat + np.eye(adj_mat.shape[0])*1.5
weight_flatten = weight_flatten.flatten()
fig, ax = plt.subplots(2,4,figsize=(20,10), sharex=True)
ax = ax.reshape((8,))
for idx, band in enumerate(filter_pool):
    if band is None:
        tdmi_data = np.load(path + '/data_series_tdmi_total.npy', allow_pickle=True)
    else:
        tdmi_data = np.load(path + '/data_series_'+band+'_tdmi_total.npy', allow_pickle=True)
    tdmi_data_cg = np.zeros_like(tdmi_data, dtype=float)
    for i in range(tdmi_data.shape[0]):
        for j in range(tdmi_data.shape[1]):
            if tdmi_mode == 'sum':
                tdmi_data[i,j] = tdmi_data[i,j][:,:,:10].sum(2)
            elif tdmi_mode == 'max':
                tdmi_data[i,j] = tdmi_data[i,j].max(2)
                # bin counts = [673,2,53,120,81,123,110,51,47]
            else:
                raise ValueError('Invalid tdmi_mode')
            if i != j:
                tdmi_data_cg[i,j]=tdmi_data[i,j].mean()
            else:
                if data_package['multiplicity'][i] > 1:
                    tdmi_data_cg[i,j]=np.mean(tdmi_data[i,j][~np.eye(data_package['multiplicity'][i], dtype=bool)])
                else:
                    tdmi_data_cg[i,j]=tdmi_data[i,j].mean()

    tdmi_data_flatten = tdmi_data_cg.flatten()
    log_tdmi_data = np.log10(tdmi_data_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    answer = weight_flatten.copy()
    # pval, cov = np.polyfit(answer.flatten(), log_tdmi_data.flatten(), deg=1,cov=True)
    # answer_set = np.unique(answer.flatten())
    # log_tdmi_data_mean = np.array([np.mean(log_tdmi_data.flatten()[answer.flatten()==key]) for key in answer_set])
    answer[answer==0]=1e-7
    log_answer = np.log10(answer)
    answer_edges = np.linspace(-7, 1, num = 10)
    log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
    for i in range(len(answer_edges)-1):
        mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
        if mask.sum() == 0:
            log_tdmi_data_mean[i] = np.nan
        else:
            log_tdmi_data_mean[i] = log_tdmi_data[mask].mean()
    pval = np.polyfit(answer_edges[:-1][~np.isnan(log_tdmi_data_mean)], log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], deg=1)
    ax[idx].plot(answer_edges[:-1], log_tdmi_data_mean, 'k.', markersize=15, label='TDMI mean')
    ax[idx].plot(answer_edges[:-1], np.polyval(pval, answer_edges[:-1]), 'r', label='Linear Fitting')
    if tdmi_mode == 'sum':
        ax[idx].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
    elif tdmi_mode == 'max':
        ax[idx].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
    ax[idx].set_xlabel('Weight')
    ticks = [-7, -5, -3, -1]
    labels = ['$10^{%d}$'%item for item in ticks]
    ax[idx].set_xticks(ticks)
    ax[idx].set_xticklabels(labels)

    if band is None:
        ax[idx].set_title(f'Origin ($R^2$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval):5.3f})')
    else:
        ax[idx].set_title(f'{band:s} ($R^2$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval):5.3f})')
    ax[idx].legend(fontsize=15)
    ax[idx].grid(ls='--')

# make last subfigure invisible
ax[-1].set_visible(False)

plt.tight_layout()
plt.savefig(path + f'cg_mi-s_{tdmi_mode:s}.png')