# %%
import numpy as np
import matplotlib.pyplot as plt
from utils.core import EcogTDMI, EcogGC

# %%
data = EcogTDMI()
data.init_data('tdmi_snr_analysis/')
# data.init_data()
sc, fc = data.get_sc_fc('ch')
roi_mask = data.roi_mask.copy()

# %%
fig, ax = plt.subplots(2,4, figsize=(12,6))
ax = ax.reshape(-1)
for i, band in enumerate(data.filters):
    (counts_0, edges_0) = np.histogram(np.log10(fc[band][sc[band]==0]), bins=20)
    (counts_1, edges_1) = np.histogram(np.log10(fc[band][(sc[band]!=0)*(sc[band]<=1e-3)]), bins=20)
    (counts_2, edges_2) = np.histogram(np.log10(fc[band][sc[band]>1e-3]), bins=20)
    ax[i].plot(edges_0[1:], counts_0, ds='steps-pre', color='navy', label='No SC')
    ax[i].plot(edges_1[1:], counts_1, ds='steps-pre', color='orange', label='Weak SC')
    ax[i].plot(edges_2[1:], counts_2, ds='steps-pre', color='red', label='Strong SC')
    ax[i].set_title(band)

[ax[i].set_xlabel(r'$\log_{10}$(TDMI)', fontsize=16) for i in (4,5,6)]
[ax[i].set_ylabel('Counts', fontsize=16) for i in (0, 4)]
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].legend(handles, labels, loc=2, fontsize=16)
ax[-1].axis('off')
plt.tight_layout()
plt.savefig('TDMI_distribution.png')
# %%
cc = np.load('data/cc.npz', allow_pickle=True)
cc_shuffle = np.load('data/cc_shuffled.npz', allow_pickle=True)
off_diag_mask = ~np.eye(117, dtype=bool)
fig, ax = plt.subplots(2, 4, figsize=(20,8))
ax = ax.reshape(-1)
for i, band in enumerate(data.filters):
    ax[i].hist(cc[band][off_diag_mask][sc[band]==0], color='b', bins=100, alpha=.5, label='SC Absent')
    ax[i].hist(cc[band][off_diag_mask][sc[band]>0],  color='r', bins=100, alpha=.5, label='SC Present')
    ax[i].hist(cc_shuffle[band][off_diag_mask], color='orange', bins=200, alpha=.5, label='Base Line')
    ax[i].set_title(band, fontsize=18)
[ax[i].set_xlabel(r'$\log_{10}$(FC)', fontsize=16) for i in (4,5,6)]
[ax[i].set_ylabel('Counts', fontsize=16) for i in (0, 4)]
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].legend(handles, labels, loc=2, fontsize=16)
ax[-1].axis('off')
plt.tight_layout()
fig.savefig('CC_distribution.png')