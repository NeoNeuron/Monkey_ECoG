# %%
import numpy as np
import matplotlib.pyplot as plt
from minfo.mi_float import tdmi_omp as TDMI
from scipy.signal import detrend
from scipy.stats import ks_2samp
from GC import GC, GC_SI
# %%

# create a gaussian random white process

plt.rcParams['font.size']=20
T = 10000
x = np.zeros((2, T))
W = np.array([
    [ 0.0, 0.1],
    [-0.1, 0.0],
])
noise = np.random.randn(2, T)*.1

for i in range(T-1):
    x[:, i+1] = 0.9*x[:, i] + W @ x[:, i] + noise[:, i]

a_max = 5.0
da = 0.1
a = np.arange(0, a_max+da, da)
gc12 = np.zeros_like(a)
gc21 = np.zeros_like(a)
cc = np.zeros_like(a)
trend = np.arange(T)/T
trend -= trend.mean()
for i in range(len(gc12)):
    # add trend
    x_trend = x.copy()
    x_trend[0,:]-= trend*da*i
    x_trend[1,:]-= trend*da*i

    gc12[i] = GC(x_trend[1,:], x_trend[0,:], order = 1)
    gc21[i] = GC(x_trend[0,:], x_trend[1,:], order = 1)
    cc[i] = np.corrcoef(x_trend)[0,1]

fig = plt.figure(figsize=(10,8))
plt.semilogy(a, gc12, label='GC 1->2', color='r')
plt.semilogy(a, gc21, label='GC 2->1', color='orange')
plt.semilogy(a, np.abs(cc), label='CC', color='navy')
plt.axhline(GC_SI(0.001, 1, T), label='GC Threshold', color='cyan')
plt.xlabel(r'$\beta$')
plt.ylabel('GC/CC Value')
plt.legend(fontsize=16)
# %%
fig.savefig('gccc_gauss_opp_trend_rec-pos.png')
# %%
fig, ax = plt.subplots(1,1, figsize=(15,8))
for i in range(2):
    ax.plot(x[i], alpha=1, label=f"neuron {i:d}")

for i in range(2):
    ax.plot(x_trend[i], alpha=1, label=f"neuron {i:d} + trend")
ax.legend(fontsize=20)
ax.set_xlim(0,T)
ax.set_xlabel('Time')
ax.set_ylabel('X')

# print(f"GC 1 -> 2 : {GC(x[1,:], x[0,:], order = 1):5.2e}")
# print(f"GC 2 -> 1 : {GC(x[0,:], x[1,:], order = 1):5.2e}")

# print(f"GC 1 -> 2 : {GC(x_trend[1,:], x_trend[0,:], order = 1):5.2e} (trend)")
# print(f"GC 2 -> 1 : {GC(x_trend[0,:], x_trend[1,:], order = 1):5.2e} (trend)")

# print(f"GC SI level : {GC_SI(0.001, 1, T):5.2e}")

# print(f"CC between 1 and 2 : {np.corrcoef(x)[0,1]:5.2f}")
# print(f"CC between 1 and 2 : {np.corrcoef(x_trend)[0,1]:5.2f} (trend)")

# %%
# %%
alpha=0.1
beta = np.arange(100)*0.01
Ct = 1/3
sigma=1
rho = (alpha*sigma**2+beta**2*Ct)/np.sqrt((sigma**2+beta**2*Ct)*((1+alpha**2)*sigma**2+beta**2*Ct))
plt.plot(beta, rho)
plt.axhline(alpha/np.sqrt(1+alpha**2), color='cyan', label=r"$\frac{\alpha}{\sqrt{1+\alpha^2}}$")
plt.title(r"$\rho=\frac{\alpha\sigma^2+\beta^2 C_T}{\sqrt{(\sigma^2+\beta^2 C_T)((1+\alpha^2) \sigma^2+\beta^2 C_T)}}$")
plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\rho$')
plt.savefig('cc_memoryless_gauss.png')
# %%
