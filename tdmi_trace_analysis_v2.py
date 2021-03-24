#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Pick up tdmi curve with high mean value.

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm, colors
from draw_causal_distribution_v2 import load_data

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
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
weight_log = np.log10(weight+1e-6)
weight_color = weight_log/weight_log.max()
weight_color = (weight_color-weight_color.min())/(weight_color.max()-weight_color.min())
my_colors = cm.Oranges(weight_color, alpha=0.5)


yaxis_type = 'linear'
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
for band in filter_pool:
    data = load_data(path, band)
    data_mean = data.mean(2)

    fig, ax = plt.subplots(1,1,figsize=(5,3), dpi=300)

    target_ids = []
    for i in range(1, data.shape[0]):
        for j in range(i+1, data.shape[0]):
            if data_mean[i,j]>0.2:
                plot_tdmi(ax, i, j, my_colors[i,j], yaxis_type=yaxis_type)
                target_ids.append([i,j])
    np.save('tmp/target_ids.npy', np.array(target_ids))
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
    plt.savefig(f'tmp/tdmi_trace_v2_{band:s}_{yaxis_type:s}.png')
    plt.close()