# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 0.5

if __name__ == '__main__':
    path = 'tdmi_snr_analysis/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)

    weight = data_package['adj_mat']
    weight[weight == 0] = 1e-6
    cg_mask = np.diag(multiplicity == 1).astype(bool)
    weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
    weight[cg_mask] = np.nan

    filter_pool = ['delta', 'theta', 'alpha',
                   'beta', 'gamma', 'high_gamma', 'raw']
    seperator = [-6, -5, -4, -3, -2, -1, 0]

    roc_data = np.load(path+'roc_WA_v3_cg.npy', allow_pickle=True)
    roc_data_ppv = np.load(path+'roc_WA_v3_ppv_cg.npy', allow_pickle=True)
    roc_data_roc = np.load(path+'roc_WA_v3_roc_cg.npy', allow_pickle=True)

    title_set = [
        "## $w_{ij}>10^{%d}$ " % item for item in seperator
    ]
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
    p_true = np.zeros_like(seperator, dtype=float)
    for idx, sep in enumerate(seperator):
        weight_mask = (weight > 10**sep)
        p_true[idx] = weight_mask.sum()*1.0/weight_mask.shape[0] / \
            weight_mask.shape[1]
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
               (1, 1), (1, 2)]
    counter = 0
    for band, idx in zip(filter_pool, indices):
        # plot figure
        ax[idx].plot(seperator, p_true, '-*',
                     markerfacecolor='None', color='r', label='p true')
        ax[idx].plot(seperator, roc_data[:, counter, -1], '-s',
                     markerfacecolor='None', color='orange', label='PPV(mi_th)')
        ax[idx].plot(seperator, roc_data_ppv[:, counter, -1], '-o',
                     markerfacecolor='None', color='orange', label='PPV(ppv_th)')
        # ax[idx].plot(seperator, roc_data_roc[:,counter, -1], '-*',
        #              markerfacecolor='None', color='orange', label='PPV(roc_th)')
        ax[idx].plot(seperator, roc_data[:, counter, -3], '-s',
                     markerfacecolor='None', color='navy', label='TPR(mi_th)')
        ax[idx].plot(seperator, roc_data_ppv[:, counter, -3], '-o',
                     markerfacecolor='None', color='navy', label='TPR(ppv_th)')
        # ax[idx].plot(seperator, roc_data_roc[:,counter, -3], '-*',
        #              markerfacecolor='None', color='navy', label='TPR(roc_th)')
        ax[idx].grid(ls='--')
        ax[idx].set_title(band)
        counter += 1

    # plot legend in the empty subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[-1, -1].legend(handles, labels, loc=1, fontsize=16)
    ax[-1, -1].axis('off')

    [ax[i, 0].set_ylabel('Percentage(%)') for i in (0, 1)]
    [ax[-1, i].set_xlabel(r'$\log_{10}$(Threshold)') for i in [0, 1, 2]]

    plt.tight_layout()
    plt.savefig(path + f'WA_v3_summary_cg.png')
    plt.close()
