#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Preprocessing monkey ECoG data. Doing band filtering.

import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import date
import os

today = date.today()
path = f'data_preprocessing_{today.isoformat():s}/'
if not os.path.isdir(path):
    os.makedirs(path)

data_path = 'ECoG data-ChenYuHan/'
data_package = {}

data_r = loadmat(data_path+'ECoG_126channel.mat')['data_r_awake']
chose = loadmat(data_path+'chose.mat')['chose'][0].astype(int)
con_known = loadmat(data_path+'map10.mat')['map10']
data_r2 = loadmat(data_path+'data_r2.mat')['data_r2']

print(data_r.shape)
print(chose.shape)
print(con_known.shape)
print(data_r2.shape)

data_package['data_r'] = data_r
data_package['chose'] = chose
data_package['con_known'] = con_known
data_package['data_r2'] = data_r2

#  band filter 
#  delta: 1-4 Hz
#  theta: 5-8 Hz
#  alpha: 9-12 Hz
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

# Plot original ECoG data sample trace
fig, ax = plt.subplots(1,2, figsize=(20,3))
ax[0].plot(np.arange(len(data_r))*0.001, data_r[:,0], label='channel 0')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Signal Intensity')
ax[0].legend()
ax[0].set_title('Original ECoG Data')
ax[0].set_xlim(0,24)
fftfreq = np.fft.fftfreq(len(data_r), d=0.001)
ax[1].semilogy(fftfreq[fftfreq>0], np.abs(np.fft.fft(data_r[:,0]))[fftfreq>0], label='channel 0')
ax[1].set_xlim(0,200)
ax[1].set_xlabel('Frequency Hz')
ax[1].set_ylabel('Spectrum Intensity')
ax[1].set_title('Power Spectrum')
ax[1].set_ylim(0.5e1,4e5)
plt.tight_layout()
plt.savefig(path +'data_r_raw_signal.png')

for band in filter_pool:
    data_filtered = np.zeros_like(data_r)
    for i in range(data_filtered.shape[1]):
        data_filtered[:,i] = filter(data_r[:,i], band, 1000)
    data_package['data_r_'+band] = data_filtered
    data_filtered_len = len(data_filtered)
    fig, ax = plt.subplots(1,2, figsize=(20,3))
    for i in range(5):
        ax[0].plot(np.arange(data_filtered_len)*0.001, data_filtered[:,i], label='channel '+str(i), alpha=0.5)
        ax[0].set_xlabel('Time(s)')
        ax[0].set_ylabel('Signal Intensity')
        ax[0].legend(loc=2, fontsize=8)
        ax[0].set_title(f'ECoG {band:s} band: {band_freq[band][0]:d}-{band_freq[band][1]:d} Hz')
        ax[0].set_xlim(0,24)
        fftfreq = np.fft.fftfreq(data_filtered_len, d=0.001)
        ax[1].semilogy(fftfreq[fftfreq>0], np.abs(np.fft.fft(data_filtered[:,i]))[fftfreq>0], label='channel '+str(i), alpha=0.5)
        ax[1].set_xlim(0,200)
        ax[1].set_ylim(0.5e1,4e5)
        ax[1].set_xlabel('Frequency Hz')
        ax[1].set_ylabel('Spectrum Intensity')
        ax[1].set_title('Power Spectrum')
    plt.tight_layout()
    plt.savefig(path + 'data_r_'+band+'.png')

# data_r2 case
# Plot original ECoG data sample trace
fig, ax = plt.subplots(1,2, figsize=(20,3))
ax[0].plot(np.arange(len(data_r2))*0.001, data_r2[:,0], label='channel 0')
ax[0].set_xlabel('Time(s)')
ax[0].set_ylabel('Signal Intensity')
ax[0].legend()
ax[0].set_title('Original ECoG Data')
ax[0].set_xlim(0,213)
fftfreq = np.fft.fftfreq(len(data_r2), d=0.001)
ax[1].semilogy(fftfreq[fftfreq>0], np.abs(np.fft.fft(data_r2[:,0]))[fftfreq>0], label='channel 0')
ax[1].set_xlim(0,200)
ax[1].set_xlabel('Frequency Hz')
ax[1].set_ylabel('Spectrum Intensity')
ax[1].set_title('Power Spectrum')
ax[1].set_ylim(1e0,2e6)
plt.tight_layout()
plt.savefig(path +'data_r2_raw_signal.png')

for band in filter_pool:
    data_filtered = np.zeros_like(data_r2)
    for i in range(data_filtered.shape[1]):
        data_filtered[:,i] = filter(data_r2[:,i], band, 1000)
    data_package['data_r2_'+band] = data_filtered
    data_filtered_len = len(data_filtered)
    fig, ax = plt.subplots(1,2, figsize=(20,3))
    for i in range(5):
        ax[0].plot(np.arange(data_filtered_len)*0.001, data_filtered[:,i], label='channel '+str(i), alpha=0.5)
        ax[0].set_xlabel('Time(s)')
        ax[0].set_ylabel('Signal Intensity')
        ax[0].legend(loc=2, fontsize=8)
        ax[0].set_title(f'ECoG {band:s} band: {band_freq[band][0]:d}-{band_freq[band][1]:d} Hz')
        ax[0].set_xlim(0,213)
        fftfreq = np.fft.fftfreq(data_filtered_len, d=0.001)
        ax[1].semilogy(fftfreq[fftfreq>0], np.abs(np.fft.fft(data_filtered[:,i]))[fftfreq>0], label='channel '+str(i), alpha=0.5)
        ax[1].set_xlim(0,200)
        ax[1].set_ylim(1e0,2e6)
        ax[1].set_xlabel('Frequency Hz')
        ax[1].set_ylabel('Spectrum Intensity')
        ax[1].set_title('Power Spectrum')
    plt.tight_layout()
    plt.savefig(path + 'data_r2_'+band+'.png')

np.savez(path + 'preprocessed_data.npz', **data_package)
