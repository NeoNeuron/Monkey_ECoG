# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
# %%
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight'].copy()
data_series = data_package['data_series_raw']
# %%
fft_freq = np.fft.fftfreq(data_series.shape[0], d=0.001)
psd = np.abs(np.fft.fft(data_series, axis=0))**2
mask = (fft_freq<100)*(fft_freq>1)
# %%
fig, ax = plt.subplots(12, 10, figsize=(48,27))
ax = ax.T.flatten()
for idx, axi in enumerate(ax):
    if idx < 117:
        axi.semilogy(
            fft_freq[mask], 
            gaussian_filter(psd[mask,idx],sigma=5), # smoothen plot curve
        )
        axi.set_title(f'ch{idx:d}')
    axi.axis('off')
plt.tight_layout()
fig.savefig('test_trace_fft.png')
# %%
band_freq = {
    'delta': [1,4],
    'theta':[5,8],
    'alpha':[9,12],
    'beta':[13,30], 
    'gamma':[31,100],
    'high_gamma':[55,100],
    'sub_delta':[0,1],
    'above_delta':[1,300],
}
ave_psd = {}
sum_psd = {}
for band, _range in band_freq.items():
    mask = (fft_freq>=_range[0])*(fft_freq<_range[1])
    ave_psd[band] = psd[mask,:].mean(0)
    sum_psd[band] = psd[mask,:].sum(0)
# %%
ave_psd_array = np.array([
    ave_psd[band] for band in ('sub_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma')
])
sum_psd_array = np.array([
    sum_psd[band] for band in ('sub_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma')
])
fig = plt.figure()
plt.plot(np.argmax(ave_psd_array, axis=0),'-', alpha=.5,label='mean')
plt.plot(np.argmax(sum_psd_array, axis=0),'-', alpha=.5,label='sum')
plt.legend()
plt.yticks([0,1,2,3,4], ['sub delta', 'delta', 'theta', 'alpha', 'beta',])
fig.savefig('test_channel_peak_band.png')
# %%
fig = plt.figure(figsize=(12,4))
for band in ('sub_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma'):
    plt.semilogy(ave_psd[band], label=band)
plt.xlabel('Channel Index')
plt.ylabel('PSD')
plt.legend()
plt.savefig('psd_mean.png')
# %%
fig = plt.figure(figsize=(12,4))
for band in ('sub_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma'):
    plt.semilogy(sum_psd[band], label=band)
plt.xlabel('Channel Index')
plt.ylabel('PSD')
plt.legend()
plt.savefig('psd_sum.png')

# %%
def map_stationary(bias, cmap):
    bias_color_min = bias.min()
    bias_color_range = bias.max()-bias.min()
    bias_color= (bias-bias_color_min)/bias_color_range
    my_colors = cmap(bias_color)

    loc = data_package['loc']
    fig, ax = plt.subplots(1,1,dpi=300)
    for i in range(loc.shape[0]):
        ax.plot(loc[i,0], loc[i,1], '.', color=my_colors[i], ms=15)
    ax.invert_yaxis()
    ax.axis('off')
    return fig, ax

# %%
for band in ('sub_delta', 'delta', 'theta', 'alpha', 'beta', 'gamma'):
    fig, ax= map_stationary(np.log10(ave_psd[band]), plt.cm.hot)
    ax.set_title(band, fontsize=20)
    fig.savefig(f'psd_map_mean_{band:s}.png')

# %%
