# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from draw_causal_distribution_v2 import load_data
from tdmi_delay_analysis_v1 import get_delay_matrix, get_snr_matrix
from cluster import get_cluster_id, get_sorted_mat

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
delay_mat = {}
snr_mat = {}
for band in filter_pool:
    tdmi_data = load_data(path, band)
    n_channel = tdmi_data.shape[0]
    n_delay = tdmi_data.shape[2]

    # complete the tdmi series
    # tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
    # tdmi_data_full[:,:,n_delay-1:] = tdmi_data
    # tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)

    delay_mat[band] = get_delay_matrix(path, band, force_compute=False)
    snr_mat[band] = get_snr_matrix(path, band, force_compute=False)
    snr_mat[band][np.eye(snr_mat[band].shape[0],dtype=bool)] = 0



fig, ax = plt.subplots(4,4, figsize=(12,12))

def get_th_val(mat, p=0.1):
    counts, edges = np.histogram(mat.flatten(), bins=100)
    mid_tick = edges[:-1] + edges[1]-edges[0]
    th_id = np.argmin(np.abs(np.cumsum(counts)/np.sum(counts) + p - 1))
    th_val = mid_tick[th_id]
    return th_val

# ax[0,3].plot(mid_tick, np.cumsum(counts)/np.sum(counts), lw=2)
# ax[0,3].axis('tight')
# ax[0,3].invert_yaxis()

def plt_unit(axi, mat, p, title):
    th_val = get_th_val(mat, p)
    axi.pcolormesh(mat>=th_val, cmap=plt.cm.gray)
    axi.set_title(title)


p = 0.25
th_val = get_th_val(snr_mat['raw'], p)
mask = (snr_mat['raw']>=th_val)
sorted_id = get_cluster_id(mask)

ax[0,0].pcolormesh(snr_mat['raw']>=th_val, cmap=plt.cm.gray)

ax[1,0].pcolormesh(weight==1.5, cmap=plt.cm.gray)

plt_unit(ax[0,1],get_sorted_mat(snr_mat['raw'],   sorted_id), p, 'raw')
plt_unit(ax[1,1],get_sorted_mat(snr_mat['delta'], sorted_id), p, 'delta')
plt_unit(ax[2,1],get_sorted_mat(snr_mat['theta'], sorted_id), p, 'theta')
plt_unit(ax[3,1],get_sorted_mat(snr_mat['alpha'], sorted_id), p, 'alpha')
plt_unit(ax[1,2],get_sorted_mat(snr_mat['beta'],  sorted_id), p, 'beta')
plt_unit(ax[2,2],get_sorted_mat(snr_mat['gamma'], sorted_id), p, 'gamma')
plt_unit(ax[3,2],get_sorted_mat(snr_mat['high_gamma'], sorted_id), p, 'high gamma')


[axi.invert_yaxis() for axi in ax.flatten()]
[axi.axis('scaled') for axi in ax.flatten()]

plt.tight_layout()
plt.savefig('tmp/delay_analysis_tmp2.png')
plt.close()
