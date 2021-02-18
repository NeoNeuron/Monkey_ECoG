#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.
# plot histogram of delays

import os
import numpy as np 
import matplotlib.pyplot as plt 
from tdmi_delay_analysis_v1 import get_delay_matrix

if __name__ == '__main__':
    path = "data_preprocessing_46_region/"
    data_package = np.load(path+"preprocessed_data.npz", allow_pickle=True)
    n_channel = data_package['stride'][-1]

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.04, right=0.25,
                                 top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])
    delay_mat = get_delay_matrix(path, 'raw')
    counts, edges = np.histogram(delay_mat[np.triu(np.ones((n_channel,n_channel)), 1).astype(bool)].flatten(), bins=50)
    ax.semilogy(edges[1:], counts, '-*')
    ax.set_title('raw')
    # plot bands
    gs = fig.add_gridspec(nrows=2, ncols=3, 
                          left=0.28, right=0.98, top=0.92, bottom=0.08, 
                          wspace=0.14)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for idx, band in enumerate(filter_pool):
        delay_mat = get_delay_matrix(path, band)
        
        counts, edges = np.histogram(delay_mat[np.triu(np.ones((n_channel,n_channel)), 1).astype(bool)].flatten(), bins=50)
        ax[idx].semilogy(edges[1:], counts, '-*')
        ax[idx].set_title(band)
  
    plt.savefig(path+f'tdmi_delay_matrix_hist.png')
    plt.close()