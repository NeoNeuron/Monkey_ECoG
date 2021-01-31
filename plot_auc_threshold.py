#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
import matplotlib.pyplot as plt
from draw_causal_distribution_v2 import MI_stats, ROC_curve, AUC

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

tdmi_mode = 'sum' # or max
is_interarea = False  # is inter area or not

fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
ax = ax.reshape((8,))
w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for idx, band in enumerate(filter_pool):
    # load data for target band
    if band is None:
        tdmi_data = np.load(path + '/data_series_tdmi_total.npy', allow_pickle=True)
    else:
        tdmi_data = np.load(path + '/data_series_'+band+'_tdmi_total.npy', allow_pickle=True)
    tdmi_data = MI_stats(tdmi_data, tdmi_mode)
    tdmi_data_flatten = tdmi_data[~np.eye(stride[-1], dtype=bool)]

    # setup interarea mask
    weight = data_package['weight']
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]
        log_tdmi_data = np.log10(tdmi_data_flatten[interarea_mask])
    else:
        log_tdmi_data = np.log10(tdmi_data_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # compute ROC curves for different w_threshold values
    aucs = np.zeros_like(w_thresholds)
    roc_thresholds = np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)
    for iidx, threshold in enumerate(w_thresholds):
        answer = weight_flatten.copy()
        answer = (answer>threshold).astype(bool)
        fpr, tpr = ROC_curve(answer, log_tdmi_data, roc_thresholds)
        aucs[iidx] = AUC(fpr, tpr)

    # plot dependence of AUC w.r.t w_threshold value
    ax[idx].semilogx(w_thresholds, aucs, '-*', color='navy')
    if band is None:
        ax[idx].set_title('Origin')
    else:
        ax[idx].set_title(band)
    ax[idx].grid(ls='--')


[ax[i].set_ylabel('AUC') for i in (0,4)]
[ax[i].set_xlabel('Threshold value') for i in (4,5,6)]

# make last subfigure invisible
ax[-1].set_visible(False)

plt.tight_layout()
if is_interarea:
    plt.savefig(path + f'auc-threshold_interarea_{tdmi_mode:s}.png')
else:
    plt.savefig(path + f'auc-threshold_{tdmi_mode:s}.png')