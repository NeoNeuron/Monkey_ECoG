#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm, colors
from draw_causal_distribution_v2 import load_data

def plot_tdmi(ax, i, j, color, yaxis_type='linear'):
    if i>=j:
        raise RuntimeError('ERROR: i should less than j')
    if yaxis_type == 'linear':
        ax.plot(data[i,j,:], lw=0.5, color=color, alpha=0.5)
        ax.plot(-np.arange(data.shape[2]), data[j,i,:], lw=0.5, color=color, alpha=0.5)
    elif yaxis_type == 'log':
        ax.semilogy(data[i,j,:], lw=0.5, color=color, alpha=0.5)
        ax.semilogy(-np.arange(data.shape[2]), data[j,i,:], lw=0.5, color=color, alpha=0.5)
    else:
        raise TypeError('Invalid yaxis type.')

path = "data_preprocessing_46_region_short/"
data_package = np.load(path+"preprocessed_data.npz", allow_pickle=True)
weight = data_package['weight']
weight_log = np.log10(weight+1e-6)
weight_color = weight_log/weight_log.max()
weight_range = np.array([-6, -5, -4, -3, -2, -1, 0])
weight_color = weight_range[:-1]+0.5
weight_color = (weight_color-weight_color.min())/(weight_color.max()-weight_color.min())
my_colors = cm.Oranges(weight_color)


yaxis_type = 'linear'
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]
for band in filter_pool:
    data = load_data(path, band)    
    
    fig, ax = plt.subplots(1,1,figsize=(5,3), dpi=300)

    # for i in range(1, data.shape[0]):
    #     for j in range(i+1, data.shape[0]):
    #         plot_tdmi(ax, i, j, 'gray', yaxis_type=yaxis_type)
    # ax.set_xlabel(r'Time delay $\tau$ (ms)')
    # ax.set_ylabel('Mutual Info (nats)')
    # ax.set_xlim(-data.shape[2], data.shape[2])

    # plot mean curves
    tdmi_mean = np.zeros((len(weight_range)-1, data.shape[2]*2-1))
    for i in range(len(weight_range)-1):
        mask = (weight_log >= weight_range[i]) & (weight_log < weight_range[i+1])
        mask = mask * np.triu(np.ones_like(weight_log), k=1).astype(bool)
        buffer = data[mask,:].mean(0)
        tdmi_mean[i,data.shape[2]-1:] = buffer
        buffer = data[mask.T,:].mean(0)
        tdmi_mean[i,:data.shape[2]] = np.flip(buffer)
    for i in range(len(weight_range)-1):
        if yaxis_type == 'linear':
            ax.plot(np.arange(-data.shape[2]+1, data.shape[2]),tdmi_mean[i], color=my_colors[i])
        elif yaxis_type == 'log':
            ax.semilogy(np.arange(-data.shape[2]+1, data.shape[2]),tdmi_mean[i], color=my_colors[i])
        else:
            raise TypeError('Invalid yaxis type.')

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
        plt.savefig(path+f'tdmi_trace_v3_{yaxis_type:s}.png')
    else:
        plt.savefig(path+f'tdmi_trace_v3_{band:s}_{yaxis_type:s}.png')
    plt.close()