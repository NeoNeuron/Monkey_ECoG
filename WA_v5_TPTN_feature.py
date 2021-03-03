# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
# * Key Notion:
# *   - weight matrix masked by weight threshold;
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;
# *   - plot the mean TDMI curves over different band and weight range.
from utils.cluster import get_cluster_id, get_sorted_mat
from utils.tdmi import compute_delay_matrix, compute_tdmi_full
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.5

def plt_unit2(axi, mat, sorted_id=None):
    if sorted_id is not None:
        buffer = get_sorted_mat(mat, sorted_id)
        axi.pcolormesh(buffer, cmap=plt.cm.gray)
    else:
        axi.pcolormesh(mat, cmap=plt.cm.gray)

path = 'tdmi_snr_analysis/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

# load and manipulate weight matrix
# weight = data_package['adj_mat']
# weight[weight == 0] = 1e-6
# cg_mask = np.diag(multiplicity == 1).astype(bool)
# weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
# weight[cg_mask] = np.nan
weight = data_package['weight']
weight[weight == 0] = 1e-6

# define filter pool and load tdmi data
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
tdmi_full = {band : compute_tdmi_full(tdmi_data[band]) for band in filter_pool}
# delay_mat = {}
# for band in filter_pool:
#     delay_mat[band] = compute_delay_matrix(tdmi_data[band])

title_set = [
    "## $w_{ij}=0$ ",
    "## $0 < w_{ij} \leq 10^{-4}$ ",
    "## $10^{-4} < w_{ij} \leq 10^{-2}$ ",
    "## $w_{ij} > 10^{-2}$ ",
    "## $w_{ij} > 0$ ",
]

# load tdmi_mask data
with open(path+'weight_analysis_v3.pkl', 'rb') as f:
    tdmi_mask_total = pickle.load(f)
    
fig1, ax1 = plt.subplots(7,5, figsize=(20, 20))
fig2, ax2 = plt.subplots(7,5, figsize=(20, 20))
for weight_mask, idx in zip(
    [
        weight == 1e-6,
        (weight <= 1e-4)*(weight > 1e-6),
        (weight > 1e-4)*(weight <= 1e-2),
        weight > 1e-2,
        weight > 1e-6,
    ],
    range(5)
):
    TP_mask = {band : weight_mask*tdmi_mask_total[title_set[idx]][band] for band in filter_pool}
    TN_mask = {band : (~weight_mask)*(~tdmi_mask_total[title_set[idx]][band]) for band in filter_pool}
    # [print(tdmi_full[band][TP_mask[band], :].shape) for band in filter_pool]
    for iidx, band in enumerate(filter_pool):
        if TP_mask[band].sum() > 0:
            ax1[iidx, idx].plot(np.arange(-tdmi_data[band].shape[2]+1, tdmi_data[band].shape[2]), tdmi_full[band][TP_mask[band], :].mean(0), color='r', alpha=1, )
            # place a text box in upper left in axes coords
            ax1[iidx, idx].text(0.05, 0.95, f'tau = {tdmi_full[band][TP_mask[band], :].mean(0).argmax()-1000:d} ms', fontsize=14,
                transform=ax1[iidx, idx].transAxes, verticalalignment='top')
    for iidx, band in enumerate(filter_pool):
        if TN_mask[band].sum() > 0:
            ax2[iidx, idx].plot(np.arange(-tdmi_data[band].shape[2]+1, tdmi_data[band].shape[2]), tdmi_full[band][TN_mask[band], :].mean(0), color='b', alpha=1, )
            # place a text box in upper left in axes coords
            ax2[iidx, idx].text(0.05, 0.95, f'tau = {tdmi_full[band][TN_mask[band], :].mean(0).argmax()-1000:d} ms', fontsize=14,
                transform=ax2[iidx, idx].transAxes, verticalalignment='top')

    ax1[0, idx].set_title(title_set[idx].strip('#'))
    ax2[0, idx].set_title(title_set[idx].strip('#'))

[ax1[iidx, 0].set_ylabel(f'{band:s} TDMI (nats)') for iidx, band in enumerate(filter_pool)]
[ax2[iidx, 0].set_ylabel(f'{band:s} TDMI (nats)') for iidx, band in enumerate(filter_pool)]

plt.tight_layout()
fig2.savefig(path + f'WA_v5_TN.png')
plt.close()
plt.tight_layout()
fig1.savefig(path + f'WA_v5_TP.png')
