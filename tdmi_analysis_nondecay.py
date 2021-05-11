#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

# %%
import numpy as np
import matplotlib.pyplot as plt
import utils
from minfo.mi_float import tdmi_omp as TDMI
from scipy.signal import detrend
from scipy.stats import ks_2samp
import pandas as pd
import joypy
from utils.filter import filter

def ridgeplot_detrend(time_series):
    """Ridgeline plot for single time series.

    Args:
        time_series (np.ndarray): single original time series.

    Returns:
        fig, ax: figure and axis.
    """
    N = time_series.shape[0]
    time_series_detrend = detrend(time_series)
    df_test = pd.DataFrame({
        'part1': np.hstack((time_series[:int(N/2)], time_series_detrend[:int(N/2)])),
        'part2': np.hstack((time_series[int(N/2):], time_series_detrend[int(N/2):])),
        'type': ['raw']*int(time_series.shape[0]/2)+['detrend']*int(time_series.shape[0]/2),
    })
    fig, ax = plt.subplots(1,1, figsize=(10,5), dpi=200)
    fig, ax = joypy.joyplot(df_test, by='type', alpha=.5, legend=True, ax=ax, bins=50)
    return fig, ax

# %%
path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)

data = utils.core.EcogTDMI()
data.init_data(path, fname='snr_th_kmean_tdmi.pkl')
sc, fc = data.get_sc_fc('ch')
snr_mask = data.get_snr_mask(path, 'snr_th_kmean_tdmi.pkl')

def map_stationary(bias, cmap):
    bias_color_max = np.abs(bias).max()
    bias_color= np.abs(bias)/bias_color_max
    my_colors = cmap(bias_color)

    loc = data_package['loc']
    fig, ax = plt.subplots(dpi=300)
    for i in range(loc.shape[0]):
        ax.plot(loc[i,0], loc[i,1], '.', color=my_colors[i], ms=15)
    ax.invert_yaxis()
    ax.axis('off')
    fig.savefig('brain_map_finer_classification.png')
    return fig

# %%
def ridgeplot_band(index:int):
    """Ridgeline plot for single time series.

    Args:
        time_series (np.ndarray): single original time series.

    Returns:
        fig, ax: figure and axis.
    """
    data_raw = data_package['data_series_raw'][:24000, index] 
    N = data_raw.shape[0]
    df_test = pd.DataFrame({
        'part1': data_raw[:int(N/2)],
        'part2': data_raw[int(N/2):],
        'band': ['raw']*int(data_raw.shape[0]/2),
    })
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']:
        data_band = data_package[f'data_series_{band:s}'][:24000, index] 
        df_test = df_test.append(pd.DataFrame({
            'part1': data_band[:int(N/2)],
            'part2': data_band[int(N/2):],
            'band': [band]*int(data_band.shape[0]/2),
        }), ignore_index=True)
    data_sub_delta = filter(data_raw, 'sub-delta', 1000)
    df_test = df_test.append(pd.DataFrame({
        'part1': data_sub_delta[:int(N/2)],
        'part2': data_sub_delta[int(N/2):],
        'band': ['sub_delta']*int(data_sub_delta.shape[0]/2),
    }), ignore_index=True)
    fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=200)
    fig, ax = joypy.joyplot(df_test, by='band', alpha=.5, legend=True, ax=ax, bins=50, )
    return fig, ax
# %%
# %%
fig,_ = ridgeplot_band(57)
fig.savefig('hist_banded_57.png')
# %%
fig,_ = ridgeplot_band(45)
fig.savefig('hist_banded_45.png')
# %%
def ridgeplot_band(index:int):
    """Ridgeline plot for single time series.

    Args:
        time_series (np.ndarray): single original time series.

    Returns:
        fig, ax: figure and axis.
    """
    fig, ax = plt.subplots(3,1, figsize=(10,10), dpi=200)
    data_raw = data_package['data_series_raw'][:24000, index] 
    ax[0].plot(data_raw)
    N = data_raw.shape[0]
    data_band = np.zeros_like(data_raw, dtype=np.complex128)
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma',]:
        data_band += np.fft.fft(data_package[f'data_series_{band:s}'][:24000, index])
    data_sub_delta = filter(data_raw, 'sub-delta', 1000)
    data_band += np.fft.fft(data_sub_delta)
    data_band = np.fft.ifft(data_band)
    ax[1].plot(data_band)
    ax[2].plot(data_sub_delta)
        # ax[1].plot(np.abs(data_band)[:1000])
    return fig, ax
fig,_ = ridgeplot_band(57)
fig.savefig('ch57_decompose.png')

# %%
def plot_stationary_hist_all_channel(data_package):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    time_series = (data_package['data_series_raw'][:, :])
    (counts_0, edges_0) = np.histogram(time_series[:12000].flatten(), bins=1000, density=True)
    (counts_1, edges_1) = np.histogram(time_series[12000:].flatten(), bins=1000, density=True)
    ax.plot(edges_0[1:], counts_0, ds='steps-pre', color='navy',   label='First Half')
    ax.plot(edges_1[1:], counts_1, ds='steps-pre', color='orange', label='Second Half')
    ax.set_xlabel(r'ECoG ($\mu$V)', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    ax.set_title('Distribution of among all channels')
    ax.legend(fontsize=20)
    fig.savefig('distribution_all_channels.png')
    return fig
# %%
# calculate stationary measure
def get_bias(data, ty='long'):
    if ty == 'long':
        bias = np.abs((data[:12000,:].mean(0) - data[12000:,:].mean(0)) / data.std(0))
    elif ty == 'short':
        bias = np.reshape(data[:24000,:].T, (data.shape[1], -1, 500)).mean(2).std(1)
    return bias
time_series = data_package['data_series_raw']
bias = get_bias(time_series)
# %%
plt.hist(bias, bins=50)
for id in np.argsort(bias)[-10:]:
    fig,_ = ridgeplot_detrend(data_package['data_series_raw'][:24000,id])
    fig.savefig(f'chn_distribution_{id:d}.png')

# %%
map_stationary(bias, plt.cm.hot)

non_wss = np.nonzero(np.abs(bias)>0.6)[0]
wss = np.nonzero(np.abs(bias)<0.05)[0]
# %%
for id in wss:
    fig,_ = ridgeplot_detrend(data_package['data_series_raw'][:24000,id])
    

# %%
target_ids = np.load('tmp/target_ids.npy', allow_pickle=True)
print(target_ids)
# plot all channel distribution
for id, index in enumerate(target_ids):
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

    for i in index:
        fig,_ = ridgeplot_detrend(data_package['data_series_raw'][:24000,i])
        fig.savefig(f'chn_distribution_{i:d}.png')


# %%
# trend manipulation
delays=3001
tdmi_data_trend = np.zeros((delays*2-1,))
trend = np.arange(24001)/24001
trend -= trend.mean()
trend *= 200

tdmi_data_trend[delays-1:] = TDMI(data_package['data_series_raw'][:,0]+trend,
                                  data_package['data_series_raw'][:,3]+trend,
                                  delays)
tdmi_data_trend[:delays] = np.flip(TDMI(data_package['data_series_raw'][:,3]+trend,
                                        data_package['data_series_raw'][:,0]+trend,
                                        delays))
plt.figure(figsize=(20,6))
plt.plot(np.arange(-delays+1, delays), tdmi_data_untrend, alpha = .5, label='original')
plt.plot(np.arange(-delays+1, delays), tdmi_data_trend,   alpha = .5, label='trend')

# %%
# draw ridgeline plot
df = pd.DataFrame(data_package['data_series_raw'])
fig, ax = plt.subplots(1,1, figsize=(12,30), dpi=200)
_, _ = joypy.joyplot(df, colormap=plt.cm.turbo, ax=ax)

# %%
# stationary distribution sorted by stationariness
def hist_with_bias_order(df_pooled, bias):
    order_with_bias = np.argsort(np.abs(bias))
    df_reorder = df_pooled[df_pooled['ch_id']==order_with_bias[0]]
    for idx in order_with_bias[1:]:
        df_reorder=df_reorder.append(df_pooled[df_pooled['ch_id']==idx], ignore_index=True)
    df_grouped = df_reorder.groupby('ch_id', sort=False)
    fig, ax = plt.subplots(1,1, figsize=(5,30), dpi=200)
    fig, ax = joypy.joyplot(df_grouped, column=['part1','part2'], alpha=.4, legend=True, ax=ax, bins=50)
    return fig
# %%
for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']:
    df_pooled = pd.DataFrame({
        'part1': data_package[f'data_series_{band:s}'][:12000,:].flatten(),
        'part2': data_package[f'data_series_{band:s}'][12000:24000,:].flatten(),
        'ch_id': np.tile(np.arange(117).reshape(-1, 1),(1,12000)).T.flatten(), 
    })
    ty = 'short'
    bias = get_bias(data_package[f'data_series_{band:s}'], ty=ty)
    fig = hist_with_bias_order(df_pooled, bias)
    fig.savefig(f'ordered_hist_{ty:s}_{band:s}.png')
# %%
fig, ax = plt.subplots(1,1, figsize=(5,30), dpi=200)
_, _ = joypy.joyplot(df_pooled[df_pooled['ch_id']<=40], by='ch_id', alpha=.4, legend=True, ax=ax, bins=50)
fig.savefig('channel_distribution_2parts_1.svg')
fig.savefig('channel_distribution_2parts_1.png')
fig, ax = plt.subplots(1,1, figsize=(5,30), dpi=200)
_, _ = joypy.joyplot(df_pooled[(df_pooled['ch_id']>40).values*(df_pooled['ch_id']<=80).values], by='ch_id', alpha=.4, legend=True, ax=ax, bins=50)
fig.savefig('channel_distribution_2parts_2.svg')
fig.savefig('channel_distribution_2parts_2.png')
fig, ax = plt.subplots(1,1, figsize=(5,30), dpi=200)
_, _ = joypy.joyplot(df_pooled[df_pooled['ch_id']>80], by='ch_id', alpha=.4, legend=True, ax=ax, bins=50)
fig.savefig('channel_distribution_2parts_3.svg')
fig.savefig('channel_distribution_2parts_3.png')
# %%
