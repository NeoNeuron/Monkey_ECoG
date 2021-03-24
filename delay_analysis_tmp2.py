# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from utils.tdmi import compute_delay_matrix, compute_snr_matrix, get_sparsity_threshold
from cluster import get_cluster_id, get_sorted_mat

path = 'data_preprocessing_46_region/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
delay_mat = {}
snr_mat = {}
for band in filter_pool:
    delay_mat[band] = compute_delay_matrix(tdmi_data[band])
    snr_mat[band] = compute_snr_matrix(tdmi_data[band])
    snr_mat[band][np.eye(snr_mat[band].shape[0],dtype=bool)] = 0

fig, ax = plt.subplots(4,4, figsize=(12,12))

def plt_unit(axi, mat, p, title):
    th_val = get_sparsity_threshold(mat, p)
    axi.pcolormesh(mat>=th_val, cmap=plt.cm.gray)
    axi.set_title(title)

p = 0.25
th_val = get_sparsity_threshold(snr_mat['raw'], p)
mask = (snr_mat['raw']>=th_val)
sorted_id = get_cluster_id(mask)

ax[0,0].pcolormesh(snr_mat['raw']>=th_val, cmap=plt.cm.gray)

ax[1,0].pcolormesh(weight==1.5, cmap=plt.cm.gray)

indices = [(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(0,1),]
for index, band in zip(indices, filter_pool):
    plt_unit(ax[index],get_sorted_mat(snr_mat[band], sorted_id), p, band)

[axi.invert_yaxis() for axi in ax.flatten()]
[axi.axis('scaled') for axi in ax.flatten()]

plt.tight_layout()
plt.savefig('tmp/delay_analysis_tmp2.png')
plt.close()
