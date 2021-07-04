# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
data_package=np.load('data/preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']

# %%
band = 'above_delta'
data_series = data_package[f'data_series_{band:s}']
fig, ax = plt.subplots(11,11,figsize=(40,30))
ax = ax.flatten()
for i in range(121):
  if i < 117:
    ax[i].plot(data_series[::1, i], np.sum(data_series*weight[i,:], axis=1)[::1], '.', ms=0.1)
    ax[i].set_xlabel('Raw Data')
    ax[i].set_ylabel('Predicted Data')
    ax[i].set_title(f'ch{i:d}')
    # ax[i].axis('scaled')
  else:
    ax[i].axis('off')

plt.tight_layout()
# %%
fig.savefig(f'test_dym_regression_{band:s}.png')
# %%
idx = 30
fig, ax = plt.subplots(1,2,figsize=(20,5))
T = np.arange(data_series.shape[0])*1e-3
ax[0].plot(data_series[::1, idx], color='b')
axt = ax[0].twinx()
iidx = np.flip(np.argsort(weight[idx,:]))
print(iidx[0])
axt.plot(np.sum(data_series*weight[idx,:], axis=1)[::1], color='red', label='All Components')
axt.plot((data_series*weight[idx,:])[:,iidx[0]], color='orange', label='Component 1')
axt.plot((data_series*weight[idx,:])[:,iidx[1]], color='chocolate', label='Component 2')
axt.legend()
ax[0].set_xlim(10000, 12000)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Raw Signal')
axt.set_ylabel('Predicted Signal')
ax[0].set_title(f'ID = {idx:d}')

ax[1].semilogy(np.flip(np.sort(weight[idx,:])), '-*')
ax[1].set_xlabel('Sorted Order')
ax[1].set_ylabel('Weight')

plt.tight_layout()
fig.savefig(f'test_dym_regression_example_trace_{band:s}_{idx:d}.png')
# %%
