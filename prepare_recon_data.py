# Author: Kai Chen
# Descriptions: Convert reconstructed binary matrix to .mat file by matching the raw data structure.
# %%
import numpy as np 
import pickle
from scipy.io import loadmat, savemat

with open('tdmi_snr_analysis/snr_th_kmean/recon_gap_tdmi.pkl', 'rb') as f:
    pickle.load(f)
    dat = pickle.load(f)
    roi_mask = pickle.load(f)
# %%

# for each region, data was stored as #(time points) by #(channel)
data_series = loadmat('ECoG_YuhanChen/r_c.mat')['r_c'][0]
num_region = data_series.shape[0]
nonzero_mask = []
for i in range(num_region):
    nonzero_mask.append((data_series[i] != 0).sum(0) != 0)
nonzero_mask = np.hstack(nonzero_mask)
num_channels = nonzero_mask.shape[0]
# %%
xx, yy = np.meshgrid(np.nonzero(nonzero_mask)[0], np.nonzero(nonzero_mask)[0])
xx, yy = xx.flatten(), yy.flatten()
whole_mat_buffer = np.zeros((num_channels, num_channels), dtype=bool)
w_dat = {}
for band in dat.keys():
    mat_buffer = np.zeros_like(roi_mask, dtype=bool)
    mat_buffer[roi_mask] = dat[band]
    whole_mat_buffer[xx,yy] = mat_buffer.flatten()
    w_dat[band] = np.array(np.nonzero(whole_mat_buffer)).T

savemat('w_binary_recon.mat', w_dat)
# %%
