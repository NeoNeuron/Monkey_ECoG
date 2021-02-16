#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
#   Coarse grain causal analysis across cortical regions
#   Plot AUC vs. answer threshold.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
import matplotlib.pyplot as plt
from draw_causal_distribution_v2 import ROC_curve, Youden_Index, AUC
from gc_analysis import load_data
from CG_gc_causal import CG

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
multiplicity = data_package['multiplicity']
stride = data_package['stride']

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

order = 10

fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
ax = ax.reshape((8,))
w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# create adj_weight_flatten by excluding 
#   auto-gc in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]

optimal_threshold = {}
for idx, band in enumerate(filter_pool):
    # load data for target band
    gc_data = load_data(path, band, order)
    gc_data_cg = CG(gc_data, stride, multiplicity)

    gc_data_flatten = gc_data_cg[cg_mask]
    gc_data_flatten[gc_data_flatten<=0] = 1e-5
    log_gc_data = np.log10(gc_data_flatten)
    log_gc_range = [log_gc_data.min(), log_gc_data.max()]

    aucs = np.zeros_like(w_thresholds)
    thresholds = np.linspace(*log_gc_range, 100)
    opt_th = np.zeros_like(w_thresholds)
    for iidx, threshold in enumerate(w_thresholds):
        answer = (adj_weight_flatten>threshold).astype(bool).flatten()
        fpr, tpr = ROC_curve(answer, log_gc_data, thresholds)
        opt_th[iidx] = thresholds[Youden_Index(fpr, tpr)]
        aucs[iidx] = AUC(fpr, tpr)
    ax[idx].semilogx(w_thresholds, aucs, '-*', color='navy')
    ax[idx].set_title(band)
    optimal_threshold[band] = opt_th
    ax[idx].grid(ls='--')

# save optimal threshold computed by Youden Index
np.savez(path + f'opt_threshold_gc_{order:d}.npz', **optimal_threshold)

[ax[i].set_ylabel('AUC') for i in (0,4)]
[ax[i].set_xlabel('Threshold value') for i in (4,5,6)]

# make last subfigure invisible
ax[-1].set_visible(False)

plt.tight_layout()
plt.savefig(path + f'cg_auc-threshold_gc_{order:d}.png')