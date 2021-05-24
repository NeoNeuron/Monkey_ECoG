# %%
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
# %%
# temporal spectrum
def tpsd(data, window, stride, fs=1000):
    tpsd_ = np.zeros((int(window//2),int((data.shape[0]-window)//stride)))
    fft_freq = np.fft.fftfreq(window, 1./fs)
    for i in range(tpsd_.shape[1]):
        fft_buffer = np.abs(np.fft.fft(data[i*stride:i*stride+window]))**2
        tpsd_[:,i]=fft_buffer[fft_freq>=0]
    time_bin_ = np.arange(tpsd_.shape[1])*stride/fs
    freq_bin_ = np.arange(tpsd_.shape[0])*fs/window
    return tpsd_, time_bin_, freq_bin_

# %%
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight'].copy()
data_series = data_package['data_series_raw']
# %%
fs = 1000
window = 2000
stride = 10
freq_max = 100 # Hz
freq_max_id = int(freq_max*window/fs)
ch_ids = np.arange(117)
tpsd_data_total = []
for ch_id in ch_ids:
    tpsd_data, time_bin, freq_bin = tpsd(data_series[:,ch_id], window, stride, fs)
    tpsd_data = gaussian_filter(tpsd_data, sigma=5)
    # tpsd_data /= tpsd_data.max()
    tpsd_data_total.append(tpsd_data)
    plt.figure(figsize=(10,5))
    xx, yy = np.meshgrid(time_bin, freq_bin[:freq_max_id])
    plt.pcolormesh(xx,yy,10*np.log10(tpsd_data[:freq_max_id,:]), cmap=plt.cm.jet)
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Ch{ch_id:d}')
    cax = plt.colorbar()
    cax.set_label('dB', loc='top', rotation=0)
    # plt.ylim(0,100)
    plt.savefig(f'test_trace_tpsd_{ch_id:d}.png')
# %%
t0 = time.time()
# prepare image container
video_duration = 5000	# ms
frame_interval = 100
nframe = 117
frame_interval_time = int(video_duration/nframe)

x_range = 2000
xmax = x_range
dx = x_range / 2

tpsd_data_total_buffer = tpsd_data_total/tpsd_data_total.max()
ims = []
fig, ax = plt.subplots(1,1, figsize=(10,8))
for i in range(117):
    xx,yy = np.meshgrid(freq_bin[:freq_max_id], np.arange(117))
    im = ax.pcolormesh(xx, yy, np.log10(tpsd_data_total_buffer[:,:freq_max_id,i]), cmap=plt.cm.jet, vmax=np.log10(tpsd_data_total_buffer.max()), vmin=np.log10(tpsd_data_total_buffer.min()), animated=True)
    if i == 0:
        ax.pcolormesh(xx, yy, np.log10(tpsd_data_total_buffer[:,:freq_max_id,i]), cmap=plt.cm.jet, vmax=np.log10(tpsd_data_total_buffer.max()), vmin=np.log10(tpsd_data_total_buffer.min()))
        ax.set_ylabel('Channel ID', fontsize=16)
        ax.set_xlabel('Freqency (Hz)', fontsize=16)
    ims.append([im])

# ani = FuncAnimation(fig, animate, interval=frame_interval_time, frames=nframe)
ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

ani.save(f"test.mp4")

print(f'generating animation takes {time.time()-t0:.3f} s')
