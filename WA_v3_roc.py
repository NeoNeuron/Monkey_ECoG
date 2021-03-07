# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;
from utils.tdmi import MI_stats
from utils.cluster import get_cluster_id, get_sorted_mat
from utils.tdmi import compute_delay_matrix, compute_snr_matrix
from utils.roc import ROC_matrix
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


if __name__ == '__main__':
    import pickle
    path = 'tdmi_snr_analysis/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    weight = data_package['weight']
    weight[weight == 0] = 1e-6
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    adj_weight = data_package['adj_mat'] + \
        np.eye(data_package['adj_mat'].shape[0])*1.5
    # cg_mask = np.diag(multiplicity == 1).astype(bool)

    # weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
    filter_pool = ['delta', 'theta', 'alpha',
                   'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data = np.load(path+'tdmi_data_long.npz', allow_pickle=True)
    delay_mat = {}
    snr_mat = {}
    seperator = [-6, -5, -4, -3, -2, -1, 0]
    for band in filter_pool:
        delay_mat[band] = compute_delay_matrix(tdmi_data[band])
        snr_mat[band] = compute_snr_matrix(tdmi_data[band])
        snr_mat[band][np.eye(snr_mat[band].shape[0], dtype=bool)] = 0

    # manually set snr threshold
    with open(path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)

    with open(path+'opt_threshold_channel_tdmi_max.pkl', 'rb') as f:
        opt_th = pickle.load(f)

    snr_mask = {}
    for band in filter_pool:
        snr_mask[band] = snr_mat[band] > snr_th[band]

    title_set = [
        "## $w_{ij}>10^{%d}$ " % item for item in seperator
    ]
    tdmi_mask_total = {}
    with open(path + 'WA_v3_roc.md', 'w') as ofile:
        roc_data = np.zeros((len(seperator), 7, 8,))
        for weight_mask, idx in zip(
            [
                weight > 10**item for item in seperator
            ],
            range(len(seperator))
        ):

            print(title_set[idx], file=ofile)
            print(f'p = {weight_mask.sum()/weight_mask.shape[0]/weight_mask.shape[1]:6.3f}', file=ofile)
            print('| band | TP | FP | FN | TN | Corr | TPR | FPR | PPV |', file=ofile)
            print('|------|----|----|----|----|------| --- | --- | --- |', file=ofile)
            sorted_id = get_cluster_id(weight_mask)

            fig, ax = plt.subplots(4, 2, figsize=(6, 12))

            tdmi_mask = {}
            for iidx, band in enumerate(filter_pool):
                # compute TDMI statistics
                tdmi_data_band = MI_stats(tdmi_data[band], 'max')
                tdmi_mask[band] = (tdmi_data_band > opt_th[band][idx])
                tdmi_mask[band] *= snr_mask[band]   # ! apply snr_mask to tdmi_mask
                TP, FP, FN, TN = ROC_matrix(weight_mask, tdmi_mask[band])
                CORR = np.corrcoef(
                    weight_mask.flatten(), 
                    tdmi_mask[band].flatten()
                )[0, 1]
                if np.isnan(CORR):
                    CORR = 0.
                print(
                    f'| {band:s} | {TP:d} | {FP:d} | {FN:d} | {TN:d} | {CORR:6.3f} | {TP/(TP+FN):6.3f} | {FP/(FP+TN):6.3f} | {TP/(TP+FP):6.3f} |', file=ofile)
                roc_data[idx, iidx, :] = [TP,FP,FN,TN,CORR,TP/(TP+FN),FP/(FP+TN),TP/(TP+FP)]
            tdmi_mask_total[title_set[idx]] = tdmi_mask

            # plot figure
            plt_unit2(ax[0, 0], weight_mask, sorted_id)
            ax[0, 0].set_title('Weight Matrix')
            ax[0, 0].set_xticklabels([])
            indices = [(1, 0), (2, 0), (3, 0), (1, 1),
                       (2, 1), (3, 1), (0, 1), ]
            union_mask = np.zeros_like(weight, dtype=bool)
            for index, band in zip(indices, filter_pool):
                plt_unit2(ax[index], tdmi_mask[band], sorted_id)
                if band != 'raw':
                    union_mask = ~((~union_mask)*(~tdmi_mask[band]))
                ax[index].set_title(band, fontsize=16)
                ax[index].set_xticklabels([])
                ax[index].set_yticklabels([])
            CORR = np.corrcoef(weight_mask.flatten(),
                               union_mask.flatten())[0, 1]
            print(f'**CORR = {CORR:6.3f}**', file=ofile)
            [axi.invert_yaxis() for axi in ax.flatten()]
            [axi.axis('scaled') for axi in ax.flatten()]

            plt.tight_layout()
            plt.savefig(path + f'WA_v3_roc_{idx:d}.png')
            plt.close()
        
        np.save(path + 'roc_WA_v3_roc.npy', roc_data)
    with open(path + 'WA_v3_roc.pkl', 'wb') as f:
        pickle.dump(tdmi_mask_total, f)