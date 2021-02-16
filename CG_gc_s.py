#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=15
mpl.rcParams['axes.labelsize'] = 15
import matplotlib.pyplot as plt
from gc_analysis import load_data
from CG_gc_causal import CG
from plot_mi_s import Linear_R2

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
multiplicity = data_package['multiplicity']
stride = data_package['stride']

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

order = 10

# create adj_weight_flatten by excluding 
#   auto-gc in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]

fig, ax = plt.subplots(2,4,figsize=(20,10))
ax = ax.reshape((8,))
for idx, band in enumerate(filter_pool):
    # 10 order is too high for theta band
    if band == 'theta':
        order = 8
    else:
        order = 10
    # load data for target band
    gc_data = load_data(path, band, order)
    gc_data_cg = CG(gc_data, stride, multiplicity)

    gc_data_flatten = gc_data_cg[cg_mask]
    gc_data_flatten[gc_data_flatten<=0] = 1e-5
    log_gc_data = np.log10(gc_data_flatten)
    log_gc_range = [log_gc_data.min(), log_gc_data.max()]

    answer = adj_weight_flatten.copy()
    answer[answer==0]=1e-7
    log_answer = np.log10(answer)
    answer_edges = np.linspace(-6, 1, num = 15)
    log_gc_data_mean = np.zeros(len(answer_edges)-1)
    for i in range(len(answer_edges)-1):
        mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
        if mask.sum() == 0:
            log_gc_data_mean[i] = np.nan
        else:
            log_gc_data_mean[i] = log_gc_data[mask].mean()
    pval = np.polyfit(answer_edges[:-1][~np.isnan(log_gc_data_mean)], log_gc_data_mean[~np.isnan(log_gc_data_mean)], deg=1)
    ax[idx].plot(answer_edges[:-1], log_gc_data_mean, 'k.', markersize=15, label='GC mean')
    ax[idx].plot(answer_edges[:-1], np.polyval(pval, answer_edges[:-1]), 'r', label='Linear Fitting')
    ticks = [-5, -3, -1]
    labels = ['$10^{%d}$'%item for item in ticks]
    ax[idx].set_xticks(ticks)
    ax[idx].set_xticklabels(labels)
    ax[idx].set_title(f'{band:s} ($r$ = {Linear_R2(answer_edges[:-1], log_gc_data_mean, pval)**0.5:5.3f})')
    ax[idx].legend(fontsize=15)
    ax[idx].grid(ls='--')

[ax[i].set_ylabel(r'$log_{10}\left((GC)\right)$') for i in (0,4)]
[ax[i].set_xlabel('Weight') for i in (4,5,6)]

# make last subfigure invisible
ax[-1].set_visible(False)

plt.tight_layout()
plt.savefig(path + f'cg_gc-s_{order:d}.png')