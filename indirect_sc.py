# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
from utils.core import EcogTDMI, EcogGC
from utils.utils import Linear_R2
def linear_fit(x, y):
    not_nan_mask = ~np.isnan(x)*~np.isnan(y)
    pval = np.polyfit(x[not_nan_mask], y[not_nan_mask], deg=1)
    r = Linear_R2(x, y, pval)**0.5
    return pval, r

# %%
data = EcogTDMI()
data.init_data('tdmi_snr_analysis/')
# data.init_data()
sc, fc = data.get_sc_fc('ch')
roi_mask = data.roi_mask.copy()

# Construct second order indirect connectivity matrix
smat = np.zeros_like(roi_mask, dtype=float)
smat[roi_mask] = sc['raw']
smat_bin = smat != 0

# %%
plt.pcolormesh(np.log10(smat), cmap=plt.cm.rainbow, vmin=-6, vmax=1)
plt.gca().invert_yaxis()
plt.axis('scaled')
# %%
w_degree2 = np.zeros_like(smat)
for i in range(roi_mask.shape[0]):
    for j in range(roi_mask.shape[1]):
        if smat_bin[i,j] == 0 and i != j:
            mask = smat_bin[:, j] * smat_bin[i, :]
            w_degree2[i,j] = np.sum(smat[mask,j]*smat[i,mask])
w_degree2_flatten = w_degree2[roi_mask]
# %%
fig, ax = plt.subplots(1,2,figsize=(15,6))
pax = ax[0].pcolormesh(np.log10(smat), cmap=plt.cm.rainbow, vmin=-6, vmax=1)
ax[0].set_title('Direct SCs')
ax[1].pcolormesh(np.log10(w_degree2), cmap=plt.cm.rainbow, vmin=-6, vmax=1)
ax[1].set_title('Indirect SCs')
[axi.invert_yaxis() for axi in ax]
[axi.axis('scaled') for axi in ax]
plt.colorbar(pax, ax=ax[1])
plt.tight_layout()
fig.savefig('Direct_Indirect_adj_mat.png')
# %%
fig, ax = plt.subplots(2,4, figsize=(12,6))
ax = ax.reshape(-1)
for i, band in enumerate(data.filters):
    (counts_1, edges_1) = np.histogram(np.log10(fc[band][w_degree2_flatten>0]), bins=20)
    (counts_2, edges_2) = np.histogram(np.log10(fc[band][sc[band]>0]), bins=20)
    ax[i].plot(edges_1[1:], counts_1, ds='steps-pre', color='b', label='Indirect SC')
    ax[i].plot(edges_2[1:], counts_2, ds='steps-pre', color='r', label='Direct SC')
    ax[i].set_title(band)

[ax[i].set_xlabel(r'$\log_{10}$(TDMI)', fontsize=16) for i in (4,5,6)]
[ax[i].set_ylabel('Counts', fontsize=16) for i in (0, 4)]
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].legend(handles, labels, loc=2, fontsize=16)
ax[-1].axis('off')
plt.tight_layout()
fig.savefig('Direct_Indirect_MI_distribution.png')
# %%
new_sc = sc['raw']+w_degree2_flatten
fig, ax = plt.subplots(1,1,figsize=(10,8))
pval,r1 = linear_fit(np.log10(new_sc), np.log10(fc['raw']))
ax.plot(np.log10(new_sc), np.log10(fc['raw']), '.b', markersize=1)
ax.plot(np.arange(-6,1), np.polyval(pval, np.arange(-6,1)), 'b', label='Fit all points')
ax.plot(np.log10(w_degree2_flatten[w_degree2_flatten>0]), np.log10(fc['raw'][w_degree2_flatten>0]), '.r', markersize=1)
pval,r2 = linear_fit(np.log10(w_degree2_flatten[w_degree2_flatten>0]), np.log10(fc['raw'][w_degree2_flatten>0]))
ax.plot(np.arange(-6,1), np.polyval(pval, np.arange(-6,1)), 'r', label='Fit indirect pairs')
ax.set_xlabel(r'$\log_{10}$(SCs)')
ax.set_ylabel(r'$\log_{10}$(TDMIs)')
ax.set_title(f'r(blue) = {r1:6.3f}, r(red) = {r2:6.3f}')
ax.legend()
plt.savefig('Direct_Indirect_sc_fc_fitting2color.png')
# %%
