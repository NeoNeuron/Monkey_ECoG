#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Convert npy data format to MATLAB format

from scipy.io import savemat
import numpy as np
path = 'data/'
data = np.load(path + 'preprocessed_data.npz')
savemat(path+'preprocessed_data.mat', data)
