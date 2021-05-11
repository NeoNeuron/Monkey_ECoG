# %%
import numpy as np
import matplotlib.pyplot as plt
from minfo.mi_float import tdmi_omp as TDMI
from scipy.signal import detrend
from scipy.stats import ks_2samp
from filter import filter

# %%
def tdcc(x, y, n_delay):
    L = x.shape[0]
    # shuffle data
    cc_value = np.zeros(n_delay)
    for i in range(n_delay):
        cc_value[i] = np.corrcoef(y[i:], x[:L-i])[0,1]
    return cc_value
# %%
# create a gaussian random white process

plt.rcParams['font.size']=20
T = 10000
x = np.zeros((2, T))
W = np.array([
    [ 0.0,-0.3],
    [+0.2, 0.0],
])
noise = np.random.randn(2, T)*.1

for i in range(T-1):
    x[:, i+1] = 0.9*x[:, i] + W @ x[:, i] + noise[:, i]


fig=plt.figure(figsize=(15,16))
gs = fig.add_gridspec(nrows=1, ncols=1, 
    left=0.05, right=0.96, top=0.95, bottom=0.72, 
    # wspace=0.15,
)
ax = fig.add_subplot(gs[0])
for i in range(2):
    ax.plot(x[i], alpha=1, label=f"neuron {i:d}")

# add trend
trend = np.sin(np.arange(T)/T*2*np.pi)
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
    # wspace=0.15,
)
ax = np.array([fig.add_subplot(i) for i in gs])
H,_,_ = np.histogram2d(x_trend[0,50:],x_trend[1,:-50],bins=50)
ax[0].pcolormesh(H)
ax[0].axis('scaled')
ax[0].axis('off')
ax[0].set_xlabel('Y')
ax[0].set_ylabel('X')
ax[0].set_title(r'$\tau$=-50')
H,_,_ = np.histogram2d(x_trend[0,:],x_trend[1,:],bins=50)
ax[1].pcolormesh(H)
ax[1].axis('scaled')
ax[1].axis('off')
ax[1].set_xlabel('Y')
ax[1].set_ylabel('X')
ax[1].set_title(r'$\tau$=0')
H,_,_ = np.histogram2d(x_trend[0,:-50],x_trend[1,50:],bins=50)
ax[2].pcolormesh(H)
ax[2].axis('scaled')
ax[2].axis('off')
ax[2].set_xlabel('Y')
ax[2].set_ylabel('X')
ax[2].set_title(r'$\tau$=50')

delays = 100
tdmi_test = np.zeros(delays*2-1)
tdmi_test[delays-1:] =       TDMI(x[0, :],x[1, :],delays)
tdmi_test[:delays] = np.flip(TDMI(x[1, :],x[0, :],delays))

tdcc_test = np.zeros(delays*2-1)
tdcc_test[delays-1:] =       tdcc(x[0, :],x[1, :],delays)
tdcc_test[:delays] = np.flip(tdcc(x[1, :],x[0, :],delays))

tdmi_test_trend = np.zeros(delays*2-1)
tdmi_test_trend[delays-1:] =       TDMI(x_trend[0, :],x_trend[1, :],delays)
tdmi_test_trend[:delays] = np.flip(TDMI(x_trend[1, :],x_trend[0, :],delays))

tdcc_test_trend = np.zeros(delays*2-1)
tdcc_test_trend[delays-1:] =       tdcc(x_trend[0, :],x_trend[1, :],delays)
tdcc_test_trend[:delays] = np.flip(tdcc(x_trend[1, :],x_trend[0, :],delays))

for i in range(x_trend.shape[0]):
    x_trend[i,:] = filter(x_trend[i,:], 'above_delta', fs=1000)
tdmi_test_trend_band = np.zeros(delays*2-1)
tdmi_test_trend_band[delays-1:] =       TDMI(x_trend[0, :],x_trend[1, :],delays)
tdmi_test_trend_band[:delays] = np.flip(TDMI(x_trend[1, :],x_trend[0, :],delays))

gs = fig.add_gridspec(nrows=1, ncols=1, 
    left=0.05, right=0.96, top=0.35, bottom=0.05, 
    # wspace=0.15, hspace=0.20,
)
ax = fig.add_subplot(gs[0])
ax.plot(np.arange(-delays+1, delays), tdmi_test, label='TDMI(raw)')
ax.plot(np.arange(-delays+1, delays), np.abs(tdcc_test), label='TDCC(raw)')
ax.plot(np.arange(-delays+1, delays), tdmi_test_trend, label='TDMI(trend)')
ax.plot(np.arange(-delays+1, delays), tdcc_test_trend, label='TDCC(trend)')
ax.plot(np.arange(-delays+1, delays), tdmi_test_trend_band, label='TDMI(trend)_filtered')
ax.set_xlabel('Delay')
ax.set_ylabel('MI')
ax.grid(ls='--')
ax.legend(fontsize=20)
# %%
fig.savefig('test_tdmi_gauss_confounder_sine_filtered.png')
# %%
