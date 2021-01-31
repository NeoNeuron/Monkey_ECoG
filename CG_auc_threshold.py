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
from draw_causal_distribution_v2 import ROC_curve, AUC
from CG_causal_distribution import Extract_MI_CG

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
multiplicity = data_package['multiplicity']
stride = data_package['stride']

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

tdmi_mode = 'sum' # or max

fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
ax = ax.reshape((8,))
w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# create adj_weight_flatten by excluding 
#   auto-tdmi in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]

for idx, band in enumerate(filter_pool):
    # load data for target band
    if band is None:
        tdmi_data = np.load(path + '/data_series_tdmi_total.npy', allow_pickle=True)
    else:
        tdmi_data = np.load(path + '/data_series_'+band+'_tdmi_total.npy', allow_pickle=True)

    tdmi_data_cg = Extract_MI_CG(tdmi_data, tdmi_mode, stride, multiplicity)

    tdmi_data_flatten = tdmi_data_cg[cg_mask]
    log_tdmi_data = np.log10(tdmi_data_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    aucs = np.zeros_like(w_thresholds)
    thresholds = np.linspace(*log_tdmi_range, 100)
    for iidx, threshold in enumerate(w_thresholds):
        answer = (adj_weight_flatten>threshold).astype(bool).flatten()
        fpr, tpr = ROC_curve(answer, log_tdmi_data, thresholds)
        aucs[iidx] = AUC(fpr, tpr)
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
plt.savefig(path + f'cg_auc-threshold_{tdmi_mode:s}.png')