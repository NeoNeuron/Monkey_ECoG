#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Convert npy data format to MATLAB format

from scipy.io import loadmat
import numpy as np
import sys
path = str(sys.argv[1])
filters = ['beta', 'gamma', 'high_gamma', 'raw']
data = {}
data_shuffle = {}
for band in filters:
    buffer = loadmat(path+f'm_data_{band:s}.mat')
    data[band] = buffer['GC']
    data_shuffle[band] = np.ones_like(buffer['GC'])*buffer['gc_zero_line'][0,0]

np.savez(path+'gc_order_6.npz', **data)
np.savez(path+'gc_shuffled_order_6.npz', **data_shuffle)