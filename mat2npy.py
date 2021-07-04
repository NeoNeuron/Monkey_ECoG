#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Convert npy data format to MATLAB format

from scipy.io import loadmat
import numpy as np
import sys
import os
import warnings
path = str(sys.argv[1])
filters = [
    'delta', 'theta', 'alpha', 'beta', 'gamma',
    'high_gamma', 'raw', 'sub_delta', 'above_delta'
]
data = {}
data_shuffle = {}
for band in filters:
    if os.path.isfile(path+f'm_data_{band:s}.mat'):
        buffer = loadmat(path+f'm_data_{band:s}.mat')
        data[band] = buffer['GC']
        data_shuffle[band] = np.ones_like(buffer['GC'])*buffer['gc_zero_line'][0,0]
    else:
        warnings.warn(f"{path:s}m_data_{band:s}.mat does not exist.")
        # data[band] = None
        # data_shuffle[band] = None

np.savez(path+'cgc.npz', **data)
np.savez(path+'cgc_shuffled.npz', **data_shuffle)