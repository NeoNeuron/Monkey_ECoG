#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm, colors

def plot_tdmi(ax, i, j, color, yaxis_type='linear'):
    if i>=j:
        raise RuntimeError('ERROR: i should less than j')
    if yaxis_type == 'linear':
        ax.plot(data[i,j,:], lw=0.5, color=color)
        ax.plot(-np.arange(data.shape[2]), data[j,i,:], lw=0.5, color=color)
    elif yaxis_type == 'log':
        ax.semilogy(data[i,j,:], lw=0.5, color=color)
        ax.semilogy(-np.arange(data.shape[2]), data[j,i,:], lw=0.5, color=color)
    else:
        raise TypeError('Invalid yaxis type.')

path = "data_preprocessing_46_region/"
data_package = np.load(path+"preprocessed_data.npz", allow_pickle=True)
weight = data_package['weight']
weight_log = np.log10(weight+1e-6)
weight_color = weight_log/weight_log.max()
weight_color = (weight_color-weight_color.min())/(weight_color.max()-weight_color.min())
my_colors = cm.Oranges(weight_color, alpha=0.5)


yaxis_type = 'log'
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]
for band in filter_pool:
    if band is None:
        try:
            data = np.load(path + f"data_series_tdmi_long_total.npy", allow_pickle=True)
        except:
            data = np.load(path + f"data_series_tdmi_total.npy", allow_pickle=True)
    else:
        try:
            data = np.load(path + f"data_series_{band:s}_tdmi_long_total.npy", allow_pickle=True)
        except:
            data = np.load(path + f"data_series_{band:s}_tdmi_total.npy", allow_pickle=True)

    fig, ax = plt.subplots(1,1,figsize=(5,3), dpi=300)

    for i in range(1, data.shape[0]):
        for j in range(i+1, data.shape[0]):
            plot_tdmi(ax, i, j, my_colors[i,j], yaxis_type=yaxis_type)
    ax.set_xlabel(r'Time delay $\tau$ (ms)')
    ax.set_ylabel('Mutual Info (nats)')
    ax.set_xlim(-data.shape[2], data.shape[2])
    # create colorbar
    norm = colors.Normalize(weight_log.min(), weight_log.max())
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.Oranges), ax=ax)
    cb.set_label('Weight',rotation=-90, verticalalignment='bottom' )
    ticks = cb.get_ticks()
    labels = ['$10^{%d}$'%item for item in ticks]
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)

    plt.tight_layout()
    plt.grid(ls='--', color='grey', lw=0.5)
    if band is None:
        plt.savefig(f'tmp/tdmi_trace_{yaxis_type:s}.png')
    else:
        plt.savefig(f'tmp/tdmi_trace_{band:s}_{yaxis_type:s}.png')
    plt.close()