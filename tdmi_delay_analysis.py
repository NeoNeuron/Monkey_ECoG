#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.

import numpy as np 

def compute_tdmi_full(tdmi_data:np.ndarray):
    n_channel = tdmi_data.shape[0]
    n_delay = tdmi_data.shape[2]
    # complete the tdmi series
    tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
    tdmi_data_full[:,:,n_delay-1:] = tdmi_data
    tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)
    return tdmi_data_full

def compute_delay_matrix(tdmi_data:np.ndarray):
    tdmi_data_full = compute_tdmi_full(tdmi_data)
    n_delay = tdmi_data.shape[2]
    delay_mat = np.argmax(tdmi_data_full, axis=2) - n_delay + 1
    return delay_mat

def compute_snr_matrix(tdmi_data:np.ndarray):
    tdmi_data_full = compute_tdmi_full(tdmi_data)
    snr_mat = (tdmi_data_full.max(2) - tdmi_data_full.mean(2))/tdmi_data_full.std(2)
    return snr_mat

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    def axis_formattor(ax):
        ax.axis('scaled')
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return ax
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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

    data_package = np.load(args.path+"preprocessed_data.npz", allow_pickle=True)
    tdmi_data = np.load(args.path+'tdmi_data.npz', allow_pickle=True)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

    # plot delay matrices
    # -------------------
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.02, right=0.25,
                                 top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])
    delay_mat = compute_delay_matrix(tdmi_data['raw'])
    pax = ax.pcolormesh(delay_mat, cmap=plt.cm.bwr)
    plt.colorbar(pax, ax=ax)
    ax.set_title('raw')
    axis_formattor(ax)
    # plot bands
    gs = fig.add_gridspec(nrows=2, ncols=3, 
                          left=0.28, right=0.98, top=0.92, bottom=0.08, 
                          wspace=0.10)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for idx, band in enumerate(filter_pool):
        delay_mat = compute_delay_matrix(tdmi_data[band])
        
        pax = ax[idx].pcolormesh(delay_mat, cmap=plt.cm.bwr)
        plt.colorbar(pax, ax=ax[idx])
        ax[idx].set_title(band)
        axis_formattor(ax[idx])

    plt.savefig(args.path+f'tdmi_delay_matrix.png')
    plt.close()

    # plot SNR matrices
    # -------------------
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.02, right=0.25,
                                 top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])
    snr_mat = compute_snr_matrix(tdmi_data['raw'])
    snr_mat[np.eye(snr_mat.shape[0], dtype=bool)] = 0

    pax = ax.pcolormesh(snr_mat, cmap=plt.cm.Oranges)
    plt.colorbar(pax, ax=ax)
    ax.set_title('raw')
    axis_formattor(ax)
    # plot bands
    gs = fig.add_gridspec(nrows=2, ncols=3, 
                          left=0.28, right=0.98, top=0.92, bottom=0.08, 
                          wspace=0.10)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for idx, band in enumerate(filter_pool):
        snr_mat = compute_snr_matrix(tdmi_data[band])
        snr_mat[np.eye(snr_mat.shape[0], dtype=bool)] = 0
        
        pax = ax[idx].pcolormesh(snr_mat, cmap=plt.cm.Oranges)
        plt.colorbar(pax, ax=ax[idx])
        ax[idx].set_title(band)
        axis_formattor(ax[idx])

    plt.savefig(args.path+f'tdmi_snr_matrix.png')
    plt.close()