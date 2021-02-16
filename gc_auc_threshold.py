#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold, GC version.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
import matplotlib.pyplot as plt
from draw_causal_distribution_v2 import ROC_curve, AUC
from gc_analysis import load_data

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

order = 10
is_interarea = False  # is inter area or not

fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
ax = ax.reshape((8,))
w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for idx, band in enumerate(filter_pool):
    # 10 order is too high for theta band
    if band == 'theta':
        order = 8
    else:
        order = 10
    # load data for target band
    gc_data = load_data(path, band, order)
    gc_data_flatten = gc_data[~np.eye(stride[-1], dtype=bool)]
    gc_data_flatten[gc_data_flatten<=0] = 1e-5

    # setup interarea mask
    weight = data_package['weight']
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]
        log_gc_data = np.log10(gc_data_flatten[interarea_mask])
    else:
        log_gc_data = np.log10(gc_data_flatten)
    log_gc_range = [log_gc_data.min(), log_gc_data.max()]

    # compute ROC curves for different w_threshold values
    aucs = np.zeros_like(w_thresholds)
    roc_thresholds = np.linspace(log_gc_range[0],log_gc_range[1],100)
    for iidx, threshold in enumerate(w_thresholds):
        answer = weight_flatten.copy()
        answer = (answer>threshold).astype(bool)
        fpr, tpr = ROC_curve(answer, log_gc_data, roc_thresholds)
        aucs[iidx] = AUC(fpr, tpr)

    # plot dependence of AUC w.r.t w_threshold value
    ax[idx].semilogx(w_thresholds, aucs, '-*', color='navy')
    ax[idx].set_title(band)
    ax[idx].grid(ls='--')


[ax[i].set_ylabel('AUC') for i in (0,4)]
[ax[i].set_xlabel('Threshold value') for i in (4,5,6)]

# make last subfigure invisible
ax[-1].set_visible(False)

plt.tight_layout()
if is_interarea:
    plt.savefig(path + f'gc_auc-threshold_interarea_{order:d}.png')
else:
    plt.savefig(path + f'gc_auc-threshold_{order:d}.png')