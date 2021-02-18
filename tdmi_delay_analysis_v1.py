#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.

import os
import numpy as np 
import matplotlib.pyplot as plt 
from draw_causal_distribution_v2 import load_data

def get_delay_matrix(path, band='raw', force_compute=True):
    fname = f'delay_matrix_{band:s}.npy'
    if os.path.isfile(path+fname) and not force_compute:
        delay_mat = np.load(path + fname, allow_pickle=True)
    else:
        tdmi_data = load_data(path, band)
        n_channel = tdmi_data.shape[0]
        n_delay = tdmi_data.shape[2]
        # complete the tdmi series
        tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
        tdmi_data_full[:,:,n_delay-1:] = tdmi_data
        tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)
        delay_mat = np.argmax(tdmi_data_full, axis=2) - n_delay + 1
        # significance test
        # not_si_mask = ((tdmi_data_full.max(2) - tdmi_data_full.mean(2)) <= 3*tdmi_data_full.std(2))
        # delay_mat[not_si_mask] = 0
        # delay_mat[np.abs(delay_mat)>=200] = 0
        np.save(path + fname, delay_mat)
    return delay_mat

def get_snr_matrix(path, band='raw', force_compute=True):
    fname = f'snr_matrix_{band:s}.npy'
    if os.path.isfile(path+fname) and not force_compute:
        snr_mat = np.load(path + fname, allow_pickle=True)
    else:
        tdmi_data = load_data(path, band)
        n_channel = tdmi_data.shape[0]
        n_delay = tdmi_data.shape[2]
        # complete the tdmi series
        tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
        tdmi_data_full[:,:,n_delay-1:] = tdmi_data
        tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)
        # significance test
        snr_mat = (tdmi_data_full.max(2) - tdmi_data_full.mean(2))/tdmi_data_full.std(2)
        np.save(path + fname, snr_mat)
    return snr_mat

def axis_formattor(ax):
    ax.axis('scaled')
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax

if __name__ == '__main__':
    path = "data_preprocessing_46_region/"
    data_package = np.load(path+"preprocessed_data.npz", allow_pickle=True)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

    # plot delay matrices
    # -------------------
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.02, right=0.25,
                                 top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])
    delay_mat = get_delay_matrix(path, 'raw')
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
        delay_mat = get_delay_matrix(path, band)
        
        pax = ax[idx].pcolormesh(delay_mat, cmap=plt.cm.bwr)
        plt.colorbar(pax, ax=ax[idx])
        ax[idx].set_title(band)
        axis_formattor(ax[idx])

    plt.savefig(path+f'tdmi_delay_matrix.png')
    plt.close()

    # plot SNR matrices
    # -------------------
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.02, right=0.25,
                                 top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])
    snr_mat = get_snr_matrix(path, 'raw')
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
        snr_mat = get_snr_matrix(path, band)
        snr_mat[np.eye(snr_mat.shape[0], dtype=bool)] = 0
        
        pax = ax[idx].pcolormesh(snr_mat, cmap=plt.cm.Oranges)
        plt.colorbar(pax, ax=ax[idx])
        ax[idx].set_title(band)
        axis_formattor(ax[idx])

    plt.savefig(path+f'tdmi_snr_matrix.png')
    plt.close()