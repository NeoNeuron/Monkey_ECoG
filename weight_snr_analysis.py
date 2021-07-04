# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from fcpy.tdmi import compute_delay_matrix, compute_snr_matrix
from fcpy.cluster import get_cluster_id, get_sorted_mat

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
# weight[weight==0] = 1e-6
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
delay_mat = {}
snr_mat = {}
for band in filter_pool:
    delay_mat[band] = compute_delay_matrix(tdmi_data[band])
    snr_mat[band] = compute_snr_matrix(tdmi_data[band])
    snr_mat[band][np.eye(snr_mat[band].shape[0],dtype=bool)] = 0
# manually set snr threshold
snr_th = {
    'raw'        :5.0,
    'delta'      :4.3,
    'theta'      :4.5,
    'alpha'      :4.,
    'beta'       :5.,
    'gamma'      :11,
    'high_gamma' :11,
}
snr_mask = {}
for band in filter_pool:
    snr_mask[band] = snr_mat[band] >= snr_th[band]

def plt_unit(axi, snr_mask, weight_mask, sorted_id=None):
    buffer = snr_mask*weight_mask
    if sorted_id is not None:
        buffer = get_sorted_mat(buffer, sorted_id)
    axi.pcolormesh(buffer, cmap=plt.cm.gray)

for weight_mask, idx in zip(
    [
        weight > 0,
        weight >= 1e-2, 
        (weight >= 1e-4)*(weight < 1e-2), 
        (weight < 1e-4)*(weight > 0),
        weight == 0
    ], 
    [0,2,4,6,8]
):
    fig, ax = plt.subplots(4,2, figsize=(6,12))

    sorted_id = get_cluster_id(weight_mask)

    plt_unit(ax[0,0], np.ones_like(weight,dtype=bool), weight_mask)
    ax[0,0].set_title('Weight Matrix')
    indices = [(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,1),]
    for index, band in zip(indices, filter_pool):
        plt_unit(ax[index], snr_mask[band], weight_mask)
        ax[index].set_title(band)

    [axi.invert_yaxis() for axi in ax.flatten()]
    [axi.axis('scaled') for axi in ax.flatten()]

    plt.tight_layout()
    plt.savefig(path + f'weight_snr_analysis_{idx:d}.png')
    plt.close()
