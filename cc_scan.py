# Author: Kai Chen
# Description: Scan functional connectivity (Correlation Coefficient) across time series.
# %%
import numpy as np
path = 'data/'
shuffle = False

data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
N = data_package['stride'][-1]

filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
# %%
cc_total = {}
for band in filters:
    data_series = data_package['data_series_'+band]
    # shuffle data
    if shuffle:
        for i in range(data_series.shape[1]):
            np.random.shuffle(data_series[:,i])
    cc_value = np.corrcoef(data_series.T)
    cc_total[band] = cc_value

# unify all data files
if shuffle:
    np.savez(path + f'cc_shuffled.npz', **cc_total)
else:
    np.savez(path + f'cc.npz', **cc_total)
