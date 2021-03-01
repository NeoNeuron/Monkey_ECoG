# Author: Kai Chen
# Description: clustering analysis of commonly activated area
#   for differet frequency band.
# * Key Notion:
# *   - weight matrix masked by weight threshold;
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from utils.tdmi import compute_delay_matrix, compute_snr_matrix, get_sparsity_threshold
from utils.tdmi import MI_stats 
from utils.cluster import get_cluster_id, get_sorted_mat
from utils.utils import CG, print_log

def plt_unit2(axi, mat, weight_mask, snr_mask, sorted_id=None):
    buffer = mat.copy()
    uni_mask = weight_mask*snr_mask
    if uni_mask.sum() == 0:
        buffer = None
    else:
        min_val = mat[weight_mask*snr_mask].min()
        buffer[~weight_mask] = min_val*0.9
        buffer[~snr_mask] = min_val*0.9
        if sorted_id is not None:
            buffer = get_sorted_mat(buffer, sorted_id)
    if buffer is None:
        axi.pcolormesh(np.zeros_like(mat), cmap=plt.cm.gray)
    elif buffer.sum() == 0:
        axi.pcolormesh(np.ones_like(mat)*weight_mask, cmap=plt.cm.gray)
    else:
        axi.pcolormesh(np.log10(buffer), cmap=plt.cm.gray)


path = 'tdmi_snr_analysis/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
weight[weight==0] = 1e-6
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
delay_mat = {}
snr_mat = {}
seperator = [-6, -4, -2, 1]
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
pval = np.load(path+'pval.npz', allow_pickle=True)

snr_mask = {}
tdmi_sep = {}
for band in filter_pool:
    snr_mask[band] = snr_mat[band] > snr_th[band]
    tdmi_sep[band] = np.array([10**(pval[band][0]*i + pval[band][1]) for i in seperator])
    tdmi_sep[band] = np.hstack((0, tdmi_sep[band]))

for weight_mask, idx in zip(
    [
        weight == 1e-6,
        (weight <= 1e-4)*(weight > 1e-6),
        (weight <= 1e-2)*(weight > 1e-4), 
        weight > 1e-2, 
        # weight > 1e-6,
    ], 
    range(4)
):

    sorted_id = get_cluster_id(weight_mask)

    fig, ax = plt.subplots(4,2, figsize=(6,12))

    tdmi_data_max = {}
    tdmi_data_cg = {}
    tdmi_mask = {}
    for band in filter_pool:
        # compute TDMI statistics
        tdmi_data_band = MI_stats(tdmi_data[band], 'max')
        tdmi_data_max[band] = tdmi_data_band.copy()
        
        tdmi_mask[band] = (tdmi_data_max[band] > tdmi_sep[band][idx])*(tdmi_data_max[band] <= tdmi_sep[band][idx+1])
        print(tdmi_mask[band].sum())


    plt_unit2(ax[0,0], weight, weight_mask, np.ones_like(weight,dtype=bool), sorted_id)
    ax[0,0].set_title('Weight Matrix')
    indices = [(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,1),]
    for index, band in zip(indices, filter_pool):
        plt_unit2(ax[index], tdmi_data_max[band], tdmi_mask[band], snr_mask[band], sorted_id)
        ax[index].set_title(band)

    [axi.invert_yaxis() for axi in ax.flatten()]
    [axi.axis('scaled') for axi in ax.flatten()]

    plt.tight_layout()
    plt.savefig(path + f'weight_analysis_v2_{idx:d}.png')
    plt.close()
