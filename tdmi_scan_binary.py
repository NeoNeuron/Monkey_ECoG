# %%
import numpy as np
from minfo.mi_bool import mutual_info
import time

def double2bool(data, th = 0):
    data_bin = (data>=th).astype(int)
    data_bin = np.maximum(np.vstack((np.zeros(data.shape[1]),np.diff(data_bin, axis=0))),0)
    print(f'Mean event rate: {data_bin.sum()/data_bin.shape[0]/data_bin.shape[1]*1000.:5.2f} Hz.')
    return data_bin.astype(bool)
# %%
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw', 'sub_delta', 'above_delta']
# %%
mi = {}
delay = 3001
for band in filter_pool:
    dat_bin = double2bool(data_package['data_series_'+band], 0)
    N = dat_bin.shape[0]

    mi[band] = np.zeros((dat_bin.shape[1], dat_bin.shape[1], 2*delay-1))
    t0 = time.time()
    for i in range(delay):
        mi[band][:,:,delay+i-1] = mutual_info(dat_bin[:N-i, :], dat_bin[i:, :], rowvar=False)
        mi[band][:,:,delay-i-1] = mutual_info(dat_bin[i:, :], dat_bin[:N-i, :], rowvar=False)
    print(f'>> mi_parallel takes {time.time() - t0:5.2f} s')

np.savez('data/tdmi_binary.npz', **mi)

# %%
import matplotlib.pyplot as plt
mi = np.load('data/tdmi_binary.npz', allow_pickle=True)
fig, ax = plt.subplots(2,5,figsize=(25,10))
ax = ax.flatten()
for i, band in enumerate(filter_pool):
    ax[i].hist(np.log10(mi[band].max(2)+1e-6).flatten(), bins=50)
    ax[i].set_title(band)
plt.tight_layout()
plt.savefig('data/tdmi_binary_distribution.png')

# %%
