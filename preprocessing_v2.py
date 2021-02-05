#!/Users/kchen/miniconda3/bin/python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Preprocessing monkey ECoG data. Doing band filtering.
#   version 2.0 for new data

import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

#  band filter 
#  delta: 1-4 Hz
#  theta: 5-8 Hz
#  alpha: 9-12Hz
#  beta: 13-30 Hz
#  gamma: 31-100 Hz
#  high gamma: 55-100 Hz
#  Pay specific attention to alpha and high gamma band;

from filter import filter

band_freq = {'delta': [1,4],
             'theta':[5,8],
             'alpha':[9,12],
             'beta':[13,30], 
             'gamma':[31,100],
             'high_gamma':[55,100]
            }

# filter target band
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

path = 'data_preprocessing_46_region/'
if not os.path.isdir(path):
    os.makedirs(path)

# load raw data
data_path = 'ECoG data-ChenYuHan/'
data_package = {}

# for each region, data was stored as #(time points) by #(channel)
data_series = loadmat(data_path+'r_c.mat')['r_c'][0]
num_region = data_series.shape[0]
data_series_len = data_series[0].shape[0]
for i in range(num_region):
    is_nonzero = ((data_series[i] != 0).sum(0) != 0)
    data_series[i] = data_series[i][:, is_nonzero].reshape((data_series_len, -1))

# number of time series in each area
multiplicity = np.zeros(num_region, dtype=int)
for i, item in enumerate(data_series):
    multiplicity[i] = item.shape[1]
data_package['multiplicity'] = multiplicity
stride = np.hstack((0, np.cumsum(multiplicity)))
data_package['stride'] = stride

# reshape the data_series
data_series = np.hstack(data_series)
data_package['data_series'] = data_series

# adjacent matrix
w = loadmat(data_path+'wei_r.mat')['wei_r']
adj_mat = np.zeros((num_region, num_region))
adj_mat[w[:,1].astype(int)-1, w[:,0].astype(int)-1] = w[:,2]
data_package['adj_mat'] = adj_mat

# create channel-wise connectivity matrix
weight = np.zeros((stride[-1],stride[-1]))
for i in range(stride.shape[0]-1):
    for j in range(stride.shape[0]-1):
        if i == j:
            weight[stride[i]:stride[i+1], stride[j]:stride[j+1]] = 1.5
        else:
            weight[stride[i]:stride[i+1], stride[j]:stride[j+1]] = data_package['adj_mat'][i,j]
weight[np.eye(weight.shape[0], dtype=bool)] = 0
data_package['weight'] = weight

for band in filter_pool:
    data_filtered = np.empty_like(data_series)
    for idx in range(data_filtered.shape[1]):
        data_filtered[:,idx] = filter(data_series[:,idx], band, 1000)
    data_package['data_series_'+band] = data_filtered

np.savez(path + 'preprocessed_data.npz', **data_package)

data_package = np.load(path+'preprocessed_data.npz', allow_pickle=True)

# Plot traces and power spectrums for original EcoG and filtered ones
from scipy.ndimage.filters import gaussian_filter1d
dt = 0.001
fig, ax = plt.subplots(2,1, figsize=(10,6))
ax[0].plot(np.arange(data_package['data_series'].shape[0])*dt, data_package['data_series'][:,0], label='channel 0')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Signal Intensity')
ax[0].legend()
ax[0].set_title('Original ECoG Data')
ax[0].set_xlim(0,24)
fftfreq = np.fft.fftfreq(data_package['data_series'].shape[0], d=dt)
power = np.abs(np.fft.fft(data_package['data_series'][:,0]))[fftfreq>0]
power_smooth = gaussian_filter1d(power, sigma=5)
ax[1].semilogy(fftfreq[fftfreq>0], power_smooth, label='channel 0')
ax[1].set_xlim(0,200)
ax[1].set_xlabel('Frequency Hz')
ax[1].set_ylabel('Spectrum Intensity')
ax[1].set_title('Power Spectrum')
ax[1].set_ylim(0.5e1,1e6)
plt.tight_layout()
plt.savefig(path +'data_series_raw_signal.png')

fig, ax = plt.subplots(len(filter_pool),2, figsize=(20,len(filter_pool)*3))
for idx, band in enumerate(filter_pool):
    data_filtered = data_package['data_series_'+band]
    data_filtered_len = data_filtered.shape[0]
    for i in range(5):
        ax[idx,0].plot(np.arange(data_filtered_len)*dt, data_filtered[:,stride[i]], label=f'channel {i+1:d}', alpha=0.5)
        ax[idx,0].set_xlabel('Time(s)')
        ax[idx,0].set_ylabel('Signal Intensity')
        ax[idx,0].set_title(f'ECoG {band:s} band: {band_freq[band][0]:d}-{band_freq[band][1]:d} Hz')
        ax[idx,0].set_xlim(0,24)
        fftfreq = np.fft.fftfreq(data_filtered_len, d=dt)
        power = np.abs(np.fft.fft(data_filtered[:,stride[i]]))[fftfreq>0]
        power_smooth = gaussian_filter1d(power, sigma=5)
        ax[idx, 1].semilogy(fftfreq[fftfreq>0], power_smooth, label=f'channel {i+1:d}', alpha=0.5)
        ax[idx, 1].set_xlim(0,200)
        ax[idx, 1].set_ylim(0.5e1,4e5)
        ax[idx, 1].set_xlabel('Frequency Hz')
        ax[idx, 1].set_ylabel('Spectrum Intensity')
        ax[idx, 1].set_title('Power Spectrum')
        ax[idx, 1].legend(loc=1, fontsize=10)
plt.tight_layout()
plt.savefig(path + 'filtered_data_pdf.png')

