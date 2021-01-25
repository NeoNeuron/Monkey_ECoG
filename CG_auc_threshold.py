#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
#   Coarse grain causal analysis across cortical regions
#   Plot AUC vs. answer threshold.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
import matplotlib.pyplot as plt

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

tdmi_mode = 'sum' # or max

fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
ax = ax.reshape((8,))
threshold_options = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
adj_mat = data_package['adj_mat']
for idx, band in enumerate(filter_pool):
    # setup interarea mask
    # weight_flatten = data_package['weight_flatten'].copy()
    # if is_interarea:
        # interarea_mask = (weight_flatten != 1.5)
        # weight_flatten = weight_flatten[interarea_mask]
    
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

    aucs = np.zeros_like(threshold_options)
    for iidx, threshold in enumerate(threshold_options):
        answer = adj_mat + np.eye(adj_mat.shape[0])*1.5
        answer = (answer>threshold).astype(bool).flatten()
        false_positive = np.array([np.sum((log_tdmi_data>i)*(~answer))/np.sum(~answer) for i in np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)])
        true_positive = np.array([np.sum((log_tdmi_data>i)*(answer))/np.sum(answer) for i in np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)])
        auc = -np.sum(np.diff(false_positive)*(true_positive[:-1]+true_positive[1:])/2)
        aucs[iidx] = auc
    ax[idx].semilogx(threshold_options, aucs, '-*')
    ax[idx].set_xlabel('Threshold value')
    if band is None:
        ax[idx].set_title('Origin')
    else:
        ax[idx].set_title(band)
    ax[idx].grid(ls='--')


ax[0].set_ylabel('AUC')
ax[4].set_ylabel('AUC')

# make last subfigure invisible
ax[-1].set_visible(False)

plt.tight_layout()
plt.savefig(path + f'cg_auc-threshold_{tdmi_mode:s}.png')