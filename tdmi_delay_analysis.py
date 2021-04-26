#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.

import pickle


if __name__ == '__main__':
    import numpy as np 
    import matplotlib.pyplot as plt 
    def axis_formattor(ax):
        ax.axis('scaled')
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return ax
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from utils.tdmi import compute_delay_matrix, compute_snr_matrix
    arg_default = {'path': 'data_preprocessing_46_region/'}
    parser = ArgumentParser(prog='tdmi_delay_analysis',
                            description = "Scan pair-wise maximum delay and SNR\
                                            of time delayed mutual information",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    args = parser.parse_args()

    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
    with open('tdmi_snr_analysis/snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)
    d_matrix = data_package['d_matrix']

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

    delay_th = 30
    mask = {}
    for band in filter_pool:
        delay_mat = np.abs(compute_delay_matrix(tdmi_data[band]))
        snr_mat = compute_snr_matrix(tdmi_data[band])
        snr_mat[np.eye(snr_mat.shape[0], dtype=bool)] = 0
        snr_mask = snr_mat > snr_th['raw']
        mask[band] = (delay_mat<delay_th) * snr_mask
    # plot delay matrices
    # -------------------
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.25,
                                 top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])

    delay_mat = np.abs(compute_delay_matrix(tdmi_data['raw']))
    # pax = ax.pcolormesh(delay_mat, cmap=plt.cm.bwr)
    ax.plot((d_matrix[mask['raw']]),delay_mat[mask['raw']], '.')
    # plt.colorbar(pax, ax=ax)
    ax.set_title('raw')
    ax.set_ylabel('Delay (ms)')
    ax.set_xlabel('Channel Distance')
    # ax.set_xlabel(r'Channel Distance$^{-1}$')
    # axis_formattor(ax)
    # plot bands
    gs = fig.add_gridspec(nrows=2, ncols=3, 
                          left=0.28, right=0.98, top=0.92, bottom=0.08, 
                          wspace=0.15)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for idx, band in enumerate(filter_pool[:-1]):
        delay_mat = np.abs(compute_delay_matrix(tdmi_data[band]))
        
        # pax = ax[idx].pcolormesh(delay_mat, cmap=plt.cm.bwr)
        ax[idx].plot((d_matrix[mask[band]]),delay_mat[mask[band]], '.')
        # plt.colorbar(pax, ax=ax[idx])
        ax[idx].set_title(band)
        # axis_formattor(ax[idx])
    
    [ax[i].set_xlabel('Channel Distance') for i in (3,4,5)]

    # plt.savefig(args.path+f'tdmi_delay_matrix.png')
    plt.savefig(args.path+f'tdmi_delay_dist.png')
    plt.close()