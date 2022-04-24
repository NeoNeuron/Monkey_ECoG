# %%
import numpy as np
import matplotlib.pyplot as plt
from minfo.mi_float import tdmi_omp as TDMI
from scipy.signal import detrend
from scipy.stats import ks_2samp
from fcpy.filter import filter
plt.rcParams['font.size']=20
from numba import njit, vectorize

# %%
def tdcc(x, y, n_delay):
    L = x.shape[0]
    # shuffle data
    cc_value = np.zeros(n_delay)
    for i in range(n_delay):
        cc_value[i] = np.corrcoef(y[i:], x[:L-i])[0,1]
    return cc_value

def draw_2d_hist(ax, x_trend, delay):
    T = x_trend.shape[1]
    x1, x2 = x_trend[0,:T-np.abs(delay)],x_trend[1,np.abs(delay):]
    if delay < 0:
        x1, x2 = x2, x1
    H,_,_ = np.histogram2d(x1,x2,bins=50)
    ax.pcolormesh(H)
    ax.axis('scaled')
    ax.axis('off')
    ax.set_title(r'$\tau$=%d'%delay)
    return ax
# %%
# create a gaussian random white process
@njit
def run_sim(W, T=1000, seed=0):
    T = 10000
    x = np.zeros((W.shape[1], T))
    noise = np.random.randn(W.shape[1], T)*.1

    x_buffer = x[:,0].copy()
    for i in range(T-1):
        x_buffer = 0.9*x_buffer + W @ x_buffer + noise[:, i]
        x[:,i+1] = x_buffer
    return x

W = np.array([
    [ 0.0,-0.3],
    [+0.2, 0.0],
])
# %%
x = run_sim(W, T=10000,)

# %%

fig=plt.figure(figsize=(15,16))
gs = fig.add_gridspec(nrows=1, ncols=1, 
    left=0.05, right=0.96, top=0.95, bottom=0.72, 
)
ax = fig.add_subplot(gs[0])
for i in range(2):
    ax.plot(x[i], alpha=1, label=f"neuron {i:d}")

# add trend
trend = np.arange(T)/T
# trend = np.sin(np.arange(T)/T*2*np.pi)
trend -= trend.mean()
# trend = np.random.randn(T)*.15
x_trend = x.copy()
x_trend[0,:]+= trend*1
x_trend[1,:]+= trend*1
# x_trend = np.zeros((2, T))
# noise[0,:]-= trend*1
# noise[1,:]+= trend*1
# for i in range(T-1):
#     x_trend[:, i+1] = 0.9*x_trend[:, i] + W @ x_trend[:, i] + noise[:, i]

for i in range(2):
    ax.plot(x_trend[i], alpha=1, label=f"neuron {i:d} + trend")
ax.legend(fontsize=16)
ax.set_xlim(0, T)
ax.set_xlabel('Time')
ax.set_ylabel('X')

gs = fig.add_gridspec(nrows=1, ncols=3, 
    left=0.05, right=0.96, top=0.65, bottom=0.40, 
)
ax = np.array([fig.add_subplot(i) for i in gs])
for idx, value in enumerate((-50, 0, 50)):
    ax[idx] = draw_2d_hist(ax[idx], x_trend, value)

delays = 100
tdmi_test = np.zeros(delays*2-1)
tdmi_test[delays-1:]    = TDMI(x[0, :],x[1, :],delays)
tdmi_test[delays-1::-1] = TDMI(x[1, :],x[0, :],delays)

tdcc_test = np.zeros(delays*2-1)
tdcc_test[delays-1:]    = tdcc(x[0, :],x[1, :],delays)
tdcc_test[delays-1::-1] = tdcc(x[1, :],x[0, :],delays)

tdmi_test_trend = np.zeros(delays*2-1)
tdmi_test_trend[delays-1:]    = TDMI(x_trend[0, :],x_trend[1, :],delays)
tdmi_test_trend[delays-1::-1] = TDMI(x_trend[1, :],x_trend[0, :],delays)

tdcc_test_trend = np.zeros(delays*2-1)
tdcc_test_trend[delays-1:]    = tdcc(x_trend[0, :],x_trend[1, :],delays)
tdcc_test_trend[delays-1::-1] = tdcc(x_trend[1, :],x_trend[0, :],delays)

# for i in range(x_trend.shape[0]):
#     x_trend[i,:] = filter(x_trend[i,:], 'gamma', fs=1000)
tdmi_test_trend_band = np.zeros(delays*2-1)
tdmi_test_trend_band[delays-1:]    = TDMI(x_trend[0, :],x_trend[1, :],delays)
tdmi_test_trend_band[delays-1::-1] = TDMI(x_trend[1, :],x_trend[0, :],delays)

gs = fig.add_gridspec(nrows=1, ncols=1, 
    left=0.05, right=0.96, top=0.35, bottom=0.05, 
    # wspace=0.15, hspace=0.20,
)
ax = fig.add_subplot(gs[0])
ax.plot(np.arange(-delays+1, delays), tdmi_test, label='TDMI(raw)')
# ax.plot(np.arange(-delays+1, delays), np.abs(tdcc_test), label='TDCC(raw)')
ax.plot(np.arange(-delays+1, delays), tdmi_test_trend, label='TDMI(trend)')
# ax.plot(np.arange(-delays+1, delays), tdcc_test_trend, label='TDCC(trend)')
ax.plot(np.arange(-delays+1, delays), tdmi_test_trend_band, label='TDMI(trend)_filtered')
ax.axvline(-10, color='gray', ls='--')
ax.axvline( 10, color='gray', ls='--')
ax.set_xlabel('Delay')
ax.set_ylabel('MI')
ax.grid(ls='--')
ax.legend(fontsize=20)
# %%
fig.savefig('test_tdmi_gauss_confounder_sine_filtered.png')
# %%
from scipy.optimize import curve_fit
W = np.array([
    [ 0.0,-0.4],
    [+0.4, 0.0],
])
x = run_sim(W, T=10000,)

fft_freq = np.fft.fftfreq(10000, 1e-3)
x_freq = np.abs(np.fft.fft(x[0,:]))

freq_range = (0, 100)
mask = (fft_freq>=freq_range[0]) * (fft_freq<freq_range[1])
plt.figure()
plt.plot(fft_freq[mask], x_freq[mask])
plt.xlim(0,100)

f = lambda x,a,b,tau: a*np.exp(-(x-b)/tau)
popt,_ = curve_fit(f, fft_freq[mask], x_freq[mask])
plt.plot(fft_freq[mask], f(fft_freq[mask],*popt), 'r')
plt.title(r'$\tau$ = %5.2f' % popt[-1])

fig = plt.figure(figsize=(12,4))
plt.plot(x[0,:])
plt.plot(x[1,:])

# %%

# add trend
trend = np.arange(T)/T
# trend = np.sin(np.arange(T)/T*2*np.pi)
trend -= trend.mean()
# trend = np.random.randn(T)*.15
x_trend = x.copy()
x_trend[0,:]+= trend*1
x_trend[1,:]+= trend*1
# x_trend = np.zeros((2, T))
# noise[0,:]-= trend*1
# noise[1,:]+= trend*1
# for i in range(T-1):
#     x_trend[:, i+1] = 0.9*x_trend[:, i] + W @ x_trend[:, i] + noise[:, i]

fig, ax = plt.subplots(2,3,figsize=(15,6), dpi=300)
ax = ax.reshape((-1))
delays = 100
tdmi_test = np.zeros(delays*2-1)
tdmi_test[delays-1:]    = TDMI(x[0, :],x[1, :],delays)
tdmi_test[delays-1::-1] = TDMI(x[1, :],x[0, :],delays)

tdmi_test_trend = np.zeros(delays*2-1)
tdmi_test_trend[delays-1:]    = TDMI(x_trend[0, :],x_trend[1, :],delays)
tdmi_test_trend[delays-1::-1] = TDMI(x_trend[1, :],x_trend[0, :],delays)

ax[0].plot(np.arange(-delays+1, delays), tdmi_test, label='TDMI(raw)')
ax[0].plot(np.arange(-delays+1, delays), tdmi_test_trend, label='TDMI(trend)')

for i, band in zip(np.arange(1,6), ('above_delta', 'theta', 'alpha', 'beta', 'gamma')):
    x_filtered = np.zeros_like(x_trend)
    for k in range(x_trend.shape[0]):
        x_filtered[k,:] = filter(x_trend[k,:], band, fs=1000)
    tdmi_test_trend_band = np.zeros(delays*2-1)
    tdmi_test_trend_band[delays-1:]    = TDMI(x_filtered[0, :],x_filtered[1, :],delays)
    tdmi_test_trend_band[delays-1::-1] = TDMI(x_filtered[1, :],x_filtered[0, :],delays)

    ax[i].plot(np.arange(-delays+1, delays), tdmi_test, label='TDMI(raw)')
    ax[i].plot(np.arange(-delays+1, delays), tdmi_test_trend_band, label='TDMI(trend)_filtered')
    ax[i].set_xlabel('Delay')
    ax[i].set_ylabel('MI')
    ax[i].set_title(band)
    ax[i].grid(ls='--')
    ax[i].axvline(-10, color='gray', ls='--')
    ax[i].axvline( 10, color='gray', ls='--')
    # ax[i].legend()

# %%
