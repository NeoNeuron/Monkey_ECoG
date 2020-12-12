#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Convert npy data format to MATLAB format

from scipy.io import savemat
import numpy as np
data = np.load('preprocessed_data.npz')
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
for band in filter_pool:
    buffer = {'data_r2_'+band : data['data_r2_'+band]}
    savemat('data_r2_'+band+'.mat', buffer)
