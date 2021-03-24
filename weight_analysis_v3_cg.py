# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
# * Key Notion:
# *   - weight matrix masked by weight threshold;
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;
from utils.tdmi import MI_stats
from weight_analysis_v3 import ROC_matrix
from utils.utils import CG
from utils.cluster import get_cluster_id, get_sorted_mat
from utils.tdmi import compute_delay_matrix, compute_snr_matrix
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.1


def plt_unit2(axi, mat, sorted_id=None):
    if sorted_id is not None:
        buffer = get_sorted_mat(mat, sorted_id)
        axi.pcolormesh(buffer, cmap=plt.cm.gray)
    else:
        axi.pcolormesh(mat, cmap=plt.cm.gray)


path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

# load and manipulate weight matrix
weight = data_package['adj_mat']
weight[weight == 0] = 1e-6
cg_mask = np.diag(multiplicity == 1).astype(bool)
weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
weight[cg_mask] = np.nan

# define filter pool and load tdmi data
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
delay_mat = {}
snr_mat = {}
seperator = [-6, -4, -2, 1]
for band in filter_pool:
    delay_mat[band] = compute_delay_matrix(tdmi_data[band])
    snr_mat[band] = compute_snr_matrix(tdmi_data[band])
    snr_mat[band][np.eye(snr_mat[band].shape[0], dtype=bool)] = 0

# manually set snr threshold
snr_th = {
    'raw': 5.0,
    'delta': 4.3,
    'theta': 4.5,
    'alpha': 4.,
    'beta': 5.,
    'gamma': 11,
    'high_gamma': 11,
}
pval = np.load(path+'pval.npz', allow_pickle=True)

snr_mask = {}
tdmi_sep = {}
for band in filter_pool:
    snr_mask[band] = snr_mat[band] > snr_th[band]
    tdmi_sep[band] = np.array(
        [10**(pval[band][0]*i + pval[band][1]) for i in seperator])
    tdmi_sep[band] = np.hstack((0, tdmi_sep[band]))

title_set = [
    "## $w_{ij}=0$ ",
    "## $0 < w_{ij} \leq 10^{-4}$ ",
    "## $10^{-4} < w_{ij} \leq 10^{-2}$ ",
    "## $w_{ij} > 10^{-2}$ ",
    "## $w_{ij} > 0$ ",
]
tdmi_mask_total = {}
with open(path + 'weight_analysis_v3_cg.md', 'w') as ofile:
    roc_data = np.zeros((5, 7, 5,))
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

        print(title_set[idx], file=ofile)
        print('| band | TP | FP | FN | TN | Corr |', file=ofile)
        print('|------|----|----|----|----|------|', file=ofile)
        sorted_id = get_cluster_id(weight_mask)

        fig, ax = plt.subplots(4, 2, figsize=(6, 12))

        tdmi_mask = {}
        for iidx, band in enumerate(filter_pool):
            # compute TDMI statistics
            tdmi_data_band = MI_stats(tdmi_data[band], 'max')
            # set filtered entities as numpy.nan
            tdmi_data_band[~snr_mask[band]] = np.nan
            # compute coarse-grain average
            tdmi_data_cg_band = CG(tdmi_data_band, stride)
            # apply cg mask
            tdmi_data_cg_band[cg_mask] = np.nan

            if idx+1 == tdmi_sep[band].shape[0]:
                tdmi_mask[band] = (tdmi_data_cg_band > tdmi_sep[band][1]) * \
                    (tdmi_data_cg_band <= tdmi_sep[band][-1])
            else:
                tdmi_mask[band] = (tdmi_data_cg_band > tdmi_sep[band]
                                   [idx])*(tdmi_data_cg_band <= tdmi_sep[band][idx+1])
            TP, FP, FN, TN = ROC_matrix(weight_mask, tdmi_mask[band])
            CORR = np.corrcoef(weight_mask.flatten(),
                               tdmi_mask[band].flatten())[0, 1]
            if np.isnan(CORR):
                CORR = 0.
            print(
                f'| {band:s} | {TP:d} | {FP:d} | {FN:d} | {TN:d} | {CORR:6.3f} |', file=ofile)
            roc_data[idx, iidx, :] = [TP, FP, FN, TN, CORR]
        tdmi_mask_total[title_set[idx]] = tdmi_mask

        # plot figure
        plt_unit2(ax[0, 0], weight_mask, sorted_id)
        ax[0, 0].set_title('Weight Matrix')
        indices = [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (0, 1), ]
        union_mask = np.zeros_like(weight, dtype=bool)
        for index, band in zip(indices, filter_pool):
            plt_unit2(ax[index], tdmi_mask[band], sorted_id)
            if band != 'raw':
                union_mask = ~((~union_mask)*(~tdmi_mask[band]))
            ax[index].set_title(band)
        CORR = np.corrcoef(weight_mask.flatten(), union_mask.flatten())[0, 1]
        print(f'**CORR = {CORR:6.3f}**', file=ofile)
        [axi.invert_yaxis() for axi in ax.flatten()]
        [axi.axis('scaled') for axi in ax.flatten()]

        plt.tight_layout()
        plt.savefig(path + f'weight_analysis_v3_{idx:d}_cg.png')
        plt.close()
    np.save(path + 'roc_weight_analysis_v3_cg.npz', roc_data)
with open(path + 'weight_analysis_v3_cg.pkl', 'wb') as f:
    pickle.dump(tdmi_mask_total, f)
