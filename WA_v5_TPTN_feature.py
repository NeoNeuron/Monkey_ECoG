# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
# * Key Notion:
# *   - weight matrix masked by weight threshold;
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;
# *   - plot the mean TDMI curves over different band and weight range.
from fcpy.tdmi import compute_tdmi_full
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.5

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

# load and manipulate weight matrix
weight = data_package['weight']
weight[weight == 0] = 1e-6

# define filter pool and load tdmi data
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
tdmi_full = {band : compute_tdmi_full(tdmi_data[band]) for band in filter_pool}

off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)

separator = [-4, -3, -2, -1, 0]
title_set = [
    "## $w_{ij}>10^{%d}$ " % item for item in separator
]

# load tdmi_mask data
with open(path+'WA_v3.pkl', 'rb') as f:
    tdmi_mask_total = pickle.load(f)

TP_mask = {}
FP_mask = {}
FN_mask = {}
TN_mask = {}
for sep, title in zip(separator, title_set):
    weight_mask = (weight > 10**sep)
    TP_mask[title] = {band : weight_mask*tdmi_mask_total[title][band] for band in filter_pool}
    FP_mask[title] = {band : (~weight_mask)*(tdmi_mask_total[title][band]) for band in filter_pool}
    FN_mask[title] = {band : (weight_mask)*(~tdmi_mask_total[title][band]) for band in filter_pool}
    TN_mask[title] = {band : (~weight_mask)*(~tdmi_mask_total[title][band]) for band in filter_pool}
    
fig1, ax1 = plt.subplots(7,len(separator), figsize=(20, 20))
fig2, ax2 = plt.subplots(7,len(separator), figsize=(20, 20))
n_curve = 100   # number of curves plot in each subplots (randomly chosen)
for mask, mask_name, color in zip(
    (TP_mask, FP_mask, FN_mask, TN_mask), 
    ('TP', 'FP', 'FN', 'TN'), 
    ('r', 'y', 'g', 'b')
):
    fig, ax = plt.subplots(7,len(separator), figsize=(20, 20))
    for idx, title in enumerate(title_set):
        for iidx, band in enumerate(filter_pool):
            if mask[title][band].sum() > 0:
                delay = np.arange(-tdmi_data[band].shape[2]+1, tdmi_data[band].shape[2])
                tdmi_full_buffer = tdmi_full[band][mask[title][band]*off_diag_mask, :]
                if tdmi_full_buffer.shape[0] < n_curve:
                    ax[iidx, idx].plot(delay, tdmi_full_buffer.T, color=color, alpha=.1, )
                else:
                    chosen_ids = np.random.choice(np.arange(tdmi_full_buffer.shape[0]), n_curve, replace=False)
                    ax[iidx, idx].plot(delay, tdmi_full_buffer[chosen_ids,:].T, color=color, alpha=.1, )
                ax[iidx, idx].set_xlim(-400,400)
                # place a text box in upper left in axes coords
                ax[iidx, idx].text(0.05, 0.95, f'tau = {tdmi_full[band][mask[title][band], :].mean(0).argmax()-3000:d} ms', fontsize=14,
                    transform=ax[iidx, idx].transAxes, verticalalignment='top')
        ax[0, idx].set_title(title.strip('#'))
    [ax[iidx, 0].set_ylabel(f'{band.upper():s} TDMI (nats)') for iidx, band in enumerate(filter_pool)]
    plt.tight_layout()
    fig.savefig(path + f'WA_v5_{mask_name:s}.png')
    plt.close()