#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.
# %%
import matplotlib.pyplot as plt 
plt.rcParams['font.size']=16
from fcpy.plot_frame import *
import numpy as np 
from fcpy.utils import Linear_R2

def plot_distance_delay(ax, delay, dist,band):
    ax.plot(delay, dist, '.')
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Channel Distance')
    pval = np.polyfit(delay, dist, deg=1)
    x = np.linspace(delay.min(), delay.max(), 10)
    ax.plot(x, np.polyval(pval, x), 'r')
    ax.set_title(band+f" r = {Linear_R2(delay, dist, pval)**0.5:6.3f}")
    return ax
# %%

# if __name__ == '__main__':
from fcpy.plot_frame import plot_union
def axis_formattor(ax):
    ax.axis('scaled')
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import utils
arg_default = {'path': 'tdmi_snr_analysis/'}
# parser = ArgumentParser(prog='tdmi_delay_analysis',
#                         description = "Scan pair-wise maximum delay and SNR\
#                                         of time delayed mutual information",
#                         formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('path', default=arg_default['path'], nargs='?',
#                     type = str, 
#                     help = "path of working directory."
#                     )
# args = parser.parse_args()

data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
d_matrix = data_package['d_matrix']
data = fcpy.core.EcogTDMI()
delay_matrix = data.get_delay_matrix()
snr_mask = data.get_snr_mask(arg_default['path'])
roi_mask = data.compute_roi_masking('ch')

delay_th = 60
mask = {}
for band in data.filters:
    delay_mat = np.abs(delay_matrix[band])
    mask[band] = (delay_mat<delay_th)
    mask[band][roi_mask] *= snr_mask[band]

data_plt = {}
for band in data.filters:
    data_plt[band] = {
        'delay':np.abs(delay_matrix[band][mask[band]]), 
        'dist':d_matrix[mask[band]],
    }
# plot delay matrices
# -------------------
fig = plot_union(data_plt, plot_distance_delay)

# plt.savefig(args.path+f'tdmi_delay_dist.png')
# plt.close()
# %%
plt.plot(data_package['loc'][:,0], data_package['loc'][:,1], '.k')
# %%
from sklearn.cluster import KMeans
N = 4
kmeans = KMeans(n_clusters=N).fit(data_package['loc'])
# if data_flatten[kmeans.labels_==0].mean() > data_flatten[kmeans.labels_==1].mean():
#     label_large, label_small = 0, 1
# else:
    # label_large, label_small = 1, 0
# kmean_th = (data_flatten[kmeans.labels_==label_large].min() + data_flatten[kmeans.labels_==label_small].max())/2
for i in range(N):
    plt.plot(data_package['loc'][kmeans.labels_==i,0], data_package['loc'][kmeans.labels_==i,1], '.', label=str(i))
plt.legend()
plt.gca().invert_yaxis()
plt.gca().axis('off')
plt.savefig('tdmi_snr_analysis/delay_analysis/brain_map.png')
# %%
for i in range(N):
    sub_module_mask = kmeans.labels_ == i
    sub_module_mask = np.outer(sub_module_mask, sub_module_mask)
    data_plt = {}
    for band in data.filters:
        data_plt[band] = {
            'delay':np.abs(delay_matrix[band][mask[band]*sub_module_mask]), 
            'dist':d_matrix[mask[band]*sub_module_mask],
            'band':band,
        }
    fig = fig_frame33(data_plt, plot_distance_delay)
    fig.savefig(f'tdmi_snr_analysis/delay_analysis/delay_distance_{i:d}.png')

# %%
sub_module_mask = np.zeros((117,117), dtype=bool)
for i in range(N):
    sub_module = kmeans.labels_ == i
    sub_module_mask += np.outer(sub_module, sub_module)
# %%
data_plt = {}
for band in data.filters:
    data_plt[band] = {
        'delay':np.abs(delay_matrix[band][mask[band]*~sub_module_mask]), 
        'dist':d_matrix[mask[band]*~sub_module_mask],
        'band':band,
    }
fig = fig_frame33(data_plt, plot_distance_delay)
fig.savefig('tdmi_snr_analysis/delay_analysis/delay_distance_interarea.png')
# %%
