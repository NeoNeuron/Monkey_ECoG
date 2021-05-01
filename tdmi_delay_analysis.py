#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.

import numpy as np 
import matplotlib.pyplot as plt 
import utils

def plot_delay_matrix(ax, data:np.ndarray):
    pax = ax.pcolormesh(data, cmap=plt.cm.bwr)
    plt.colorbar(pax, ax=ax)
    ax.axis('scaled')
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax

def plot_delay_hist(ax, data:np.ndarray, bins=100):
    ax.hist(data[np.triu(np.ones_like(data), 1).astype(bool)].flatten(), bins = bins)
    ax.set_xlabel('Time Delay(ms)')
    ax.set_ylabel('Counts')
    return ax

if __name__ == '__main__':
    from utils.plot import plot_union
    path = "data_preprocessing_46_region/"
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    data = utils.core.EcogTDMI()
    delay_matrix = data.get_delay_matrix()
    stride = data_package['stride']
    n_region = stride.shape[0]-1

    fig = plot_union(delay_matrix, plot_delay_matrix)
    plt.savefig(path+f'tdmi_delay_matrix_channel.png')
    plt.close()

    data_plt = {}
    for band in data.filters:
        # average within regions
        delay_mat_region = np.zeros((n_region, n_region))
        for i in range(n_region):
            for j in range(n_region):
                delay_mat_region[i,j] = delay_matrix[band][stride[i]:stride[i+1],stride[j]:stride[j+1]].mean()
                # if i != j:
                #     delay_mat_region[i,j]=data_buffer.mean()
                # else:
                #     if multiplicity[i] > 1:
                #         delay_mat_region[i,j]=np.mean(data_buffer[~np.eye(multiplicity[i], dtype=bool)])
                #     else:
                #         delay_mat_region[i,j]=data_buffer.mean() # won't be used in ROC.
        data_plt[band] = delay_mat_region.copy()
    binary_delay_match = {}
    for band in data.filters:
        binary_delay_match[band] = (data_plt[band]-data_plt['raw'])>=0
        print(f'{band:s} band matching ratio : {binary_delay_match[band].astype(float).mean()*100.:3.3f} %' )

    fig = plot_union(data_plt, plot_delay_matrix)
    plt.savefig(path+f'tdmi_delay_matrix_region.png')
    plt.close()

    fig = plot_union(data_plt, plot_delay_hist)
    plt.savefig(path+f'tdmi_delay_matrix_region_hist.png')
    plt.close()