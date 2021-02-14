# Author: Kai Chen
import numpy as np
from GC import GC
from tdmi_scan_v2 import print_log
import time

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']

filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
N = stride[-1]

order = 10
shuffle = False

t0 = time.time()
for band in filters:
    data_series = data_package['data_series_'+band]
    # shuffle data
    if shuffle:
        for i in range(data_series.shape[1]):
            np.random.shuffle(data_series[:,i])
    gc_value = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gc_value[i,j] = GC(data_series[:,i],data_series[:,j], order)

    if shuffle:
        print_log(band + f' shuffled GC (order {order:d}) finished', t0)
        np.save(path + f'gc_values_{band:s}_shuffled_order_{order:d}.npy', gc_value)
    else:
        print_log('raw signal' + f' GC (order {order:d}) finished', t0)
        np.save(path + f'gc_values_raw_order_{order:d}.npy', gc_value)