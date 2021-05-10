# Author: Kai Chen
# Description: Scan functional connectivity (Correlation Coefficient) across time series.
# %%
import numpy as np
from scipy.signal import detrend
path = 'data/'
shuffle = False

data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
N = data_package['stride'][-1]

filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
# %%
n_delay = 3001
cc_total = {}
for band in filters:
    data_series = data_package['data_series_'+band]
    data_series = detrend(data_package['data_series_'+band], axis=0)
    L = data_series.shape[0]
    # shuffle data
    if shuffle:
        for i in range(data_series.shape[1]):
            np.random.shuffle(data_series[:,i])
    cc_value = np.zeros((N,N,n_delay))
    for i in range(n_delay):
        cc_buffer = np.corrcoef(data_series.T[:,i:], data_series.T[:,:L-i])
        cc_value[:,:,i] = cc_buffer[:N, N:]
    cc_total[band] = cc_value
# %%
# unify all data files
if shuffle:
    np.savez(path + f'tdcc_shuffled.npz', **cc_total)
    # np.savez(path + f'tdcc_detrend_shuffled.npz', **cc_total)
else:
    np.savez(path + f'tdcc.npz', **cc_total)
    # np.savez(path + f'tdcc_detrend.npz', **cc_total)
