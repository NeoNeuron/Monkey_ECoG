#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

# %%
import numpy as np
import matplotlib.pyplot as plt
import utils
from minfo.mi_float import tdmi_omp as TDMI
from scipy.signal import detrend
from scipy.stats import ks_2samp
# %%

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)

target_ids = np.load('tmp/target_ids.npy', allow_pickle=True)
print(target_ids)
# %%
data = utils.core.EcogTDMI()
data.init_data(path, fname='snr_th_kmean_tdmi.pkl')
sc, fc = data.get_sc_fc('ch')
snr_mask = data.get_snr_mask(path, 'snr_th_kmean_tdmi.pkl')

# %%
fig, ax = plt.subplots(1,1,figsize=(10,6))
time_series = (data_package['data_series_raw'][:, 0])
(counts_0, edges_0) = np.histogram(time_series[:12000], bins=100, density=True)
(counts_1, edges_1) = np.histogram(time_series[12000:], bins=100, density=True)
ax.plot(edges_0[1:], counts_0, ds='steps-pre', color='navy',   label='First Half')
ax.plot(edges_1[1:], counts_1, ds='steps-pre', color='orange', label='Second Half')
ax.set_xlabel(r'ECoG ($\mu$V)', fontsize=20)
ax.set_ylabel('Counts', fontsize=20)
ax.set_title('Distribution of among all channels')
ax.legend(fontsize=20)
fig.savefig('distribution_all_channels.png')
# %%
print(ks_2samp(data_package['data_series_raw'][:12000:100, 0].flatten(),data_package['data_series_raw'][12000::100, 0].flatten()))
# %%
plt.plot(data_package['data_series_raw'][:12000:100, 0])
plt.plot(data_package['data_series_raw'][12000::100, 0])
# %%
# delays = 3001
# tdmi_data = np.zeros((target_ids.shape[0], delays*2-1))
# for i, idx in enumerate(target_ids):
#     tdmi_data[i,delays-1:] = data.tdmi_data['raw'][idx[0], idx[1],:]
#     tdmi_data[i,:delays] = np.flip(data.tdmi_data['raw'][idx[1], idx[0],:])

# %%
# plt.figure(figsize=(20,10))
# for i, curve in enumerate(tdmi_data[:10,:]):
#     plt.plot(curve, label=str(i))
# plt.legend()
# %%
for id, index in enumerate(target_ids):
# id = 100
# index = target_ids[2]
    delays = 3001
    fig = plt.figure(figsize=(20,20), dpi=100)
    gs = fig.add_gridspec(nrows=1, ncols=2, 
                        left=0.05, right=0.96, top=0.96, bottom=0.75, 
                        wspace=0.15)
    ax = np.array([fig.add_subplot(i) for i in gs])

    for i in range(2):
        (counts_0, edges_0) = np.histogram(data_package['data_series_raw'][:12000, index[i]], bins=100)
        (counts_1, edges_1) = np.histogram(data_package['data_series_raw'][12000:, index[i]], bins=100)
        ax[i].plot(edges_0[1:], counts_0, ds='steps-pre', color='navy',   label='First Half')
        ax[i].plot(edges_1[1:], counts_1, ds='steps-pre', color='orange', label='Second Half')
        ax[i].set_xlabel(r'ECoG ($\mu$V)', fontsize=20)
        ax[i].set_ylabel('Counts')
        ax[i].legend(fontsize=20)

    gs = fig.add_gridspec(nrows=2, ncols=1, 
                        left=0.05, right=0.96, top=0.70, bottom=0.05, 
                        hspace=0.15)
    ax = np.array([fig.add_subplot(i) for i in gs])

    for i in index:
        ax[0].plot(data_package['data_series_raw'][:, i], label=f'channel {i:d}')
        ax[0].plot(detrend(data_package['data_series_raw'][:, i]), label=f'channel {i:d}(detrend)')
        ax[0].set_xlabel('Time(ms)')
        ax[0].set_ylabel(r'ECoG ($\mu$V)', fontsize=20)
        ax[0].grid(ls='--')
    ax[0].legend(fontsize=20)


    delays = 5001
    tdmi_data_detrend = np.zeros((delays*2-1,))
    tdmi_data_detrend[delays-1:] = TDMI(detrend(data_package['data_series_raw'][:,index[1]]),
                                    detrend(data_package['data_series_raw'][:,index[0]]),
                                    delays)
    tdmi_data_detrend[:delays] = np.flip(TDMI(detrend(data_package['data_series_raw'][:,index[0]]),
                                        detrend(data_package['data_series_raw'][:,index[1]]),
                                        delays))

    tdmi_data_untrend = np.zeros((delays*2-1,))
    tdmi_data_untrend[delays-1:] = TDMI((data_package['data_series_raw'][:,index[1]]),
                                    (data_package['data_series_raw'][:,index[0]]),
                                    delays)
    tdmi_data_untrend[:delays] = np.flip(TDMI((data_package['data_series_raw'][:,index[0]]),
                                        (data_package['data_series_raw'][:,index[1]]),
                                        delays))
    ax[1].plot(np.arange(-delays+1, delays), tdmi_data_detrend, label='detrend')
    ax[1].plot(np.arange(-delays+1, delays), tdmi_data_untrend, label='untrend')
    ax[1].set_xlabel('Delay(ms)')
    ax[1].set_ylabel('MI')
    ax[1].grid(ls='--')
    ax[1].legend(fontsize=20)
    ax[1].set_title(f"channel {index[1]:d} -> {index[0]:d} : {data_package['weight'][index[0], index[1]]:5.2e},   "
                    f"channel {index[0]:d} -> {index[1]:d} : {data_package['weight'][index[1], index[0]]:5.2e}.", 
                    fontsize=16)
    fig.savefig(f'TDMI_nondecay_analysis_{id:d}.png')

# # %%
# # create a gaussian random white process

# T = 10000
# x = np.zeros((2, T))
# W = np.array([[0, 0.],[-0.1, 0]])
# noise = np.random.randn(2, T)*.1

# for i in range(T-1):
#     x[:, i+1] = -0.9*x[:, i] + W @ x[:, i] + noise[:, i]

# # add trend
# trend = np.arange(T)/T
# trend -= trend.mean()
# x[1,:]+= trend*1
# x[0,:]+= trend*.5

# plt.figure(figsize=(20,10))
# for i in range(2):
#     plt.plot(x[i], label=f"neuron {i:d}")
# plt.legend()
# # %%
# delays = 100
# tdmi_test = np.zeros((delays*2-1,))

# tdmi_test[delays-1:] = TDMI(x[0,:],
#                             x[1,:],
#                             delays)
# tdmi_test[:delays] = np.flip(TDMI(x[1, :],
#                                   x[0, :],
#                                   delays))

# # %%
# plt.figure(figsize=(20,10))
# plt.plot(np.arange(-delays+1, delays), tdmi_test)
# plt.xlabel('Delay(ms)')
# plt.ylabel('MI')
# plt.grid(ls='--')
# # %%
# delays=3001
# tdmi_data_untrend = np.zeros((delays*2-1,))

# tdmi_data_untrend[delays-1:] = TDMI(data_package['data_series_raw'][:,0],
#                                   data_package['data_series_raw'][:,3],
#                                   delays)
# tdmi_data_untrend[:delays] = np.flip(TDMI(data_package['data_series_raw'][:,3],
#                                         data_package['data_series_raw'][:,0],
#                                         delays))
# # %%

# delays=3001
# tdmi_data_trend = np.zeros((delays*2-1,))
# trend = np.arange(24001)/24001
# trend -= trend.mean()
# trend *= 200

# tdmi_data_trend[delays-1:] = TDMI(data_package['data_series_raw'][:,0]+trend,
#                                   data_package['data_series_raw'][:,3]+trend,
#                                   delays)
# tdmi_data_trend[:delays] = np.flip(TDMI(data_package['data_series_raw'][:,3]+trend,
#                                         data_package['data_series_raw'][:,0]+trend,
#                                         delays))
# plt.figure(figsize=(20,6))
# plt.plot(np.arange(-delays+1, delays), tdmi_data_untrend, alpha = .5, label='original')
# plt.plot(np.arange(-delays+1, delays), tdmi_data_trend,   alpha = .5, label='trend')
# # %%

# %%
