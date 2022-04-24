# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from fcpy.core import EcogTDMI, EcogGC
from fcpy.utils import linear_fit
# %%
data = EcogTDMI()
# data.init_data('tdmi_snr_analysis/')
data.init_data()
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
direct_sc = sc['raw'].copy()
direct_sc[direct_sc==0] = np.nan
direct_sc[direct_sc==1.5] = np.nan
new_sc = sc['raw']+w_degree2_flatten
new_sc[new_sc==1.5] = np.nan
plt.hist(np.log10(new_sc), bins=120, alpha=.5, label='all')
plt.hist(np.log10(direct_sc), bins=120, alpha=.5, label='sc direct')
plt.hist(np.log10(w_degree2_flatten[w_degree2_flatten!=0]), bins=120, alpha=.5, label='sc indirect')
plt.legend()

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
fig, ax = plt.subplots(2,5, figsize=(25,12), dpi=100)
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
fig = plt.figure(figsize=(30,14))
gs = fig.add_gridspec(nrows=2, ncols=5, 
                      left=0.06, right=0.98, top=0.92, bottom=0.10, 
                      wspace=0.24, hspace=0.30)
ax = np.array([fig.add_subplot(i) for i in gs])
for i, band in enumerate(fc.keys()):
    # new_sc = sc[band]+w_degree2_flatten
    pval1,r1 = linear_fit(np.log10(sc[band]), np.log10(fc[band]))
    ax[i].plot(np.log10(sc[band]), np.log10(fc[band]), '.b', markersize=1)
    ax[i].plot(np.arange(-6,1), np.polyval(pval1, np.arange(-6,1)), 'b', label='Fit direct pairs')
    ax[i].plot(np.log10(w_degree2_flatten[w_degree2_flatten>0]), np.log10(fc[band][w_degree2_flatten>0]), '.r', markersize=1)
    pval2,r2 = linear_fit(np.log10(w_degree2_flatten[w_degree2_flatten>0]), np.log10(fc[band][w_degree2_flatten>0]))
    ax[i].plot(np.arange(-6,1), np.polyval(pval2, np.arange(-6,1)), 'r', label='Fit indirect pairs')
    ax[i].set_title(f'r(blue) = {r1:6.3f}, r(red) = {r2:6.3f} \n k(blue) = {pval1[0]:6.3f}, k(red) = {pval2[0]:6.3f}', fontsize=19)
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].set_visible(True)
ax[-1].legend(handles, labels, loc=2, fontsize=20)
ax[-1].axis('off')
[fig.get_axes()[i].set_ylabel(r'$\log_{10}$(TDMI)') for i in (0,5)]
[fig.get_axes()[i].set_xlabel('$\log_{10}$(SC)') for i in (5,6,7,8)]
plt.savefig('Direct_Indirect_sc_fc_fitting2color_mi.png')

# %%
fig = plt.figure(figsize=(30,14))
gs = fig.add_gridspec(nrows=2, ncols=5, 
                      left=0.06, right=0.98, top=0.92, bottom=0.10, 
                      wspace=0.24, hspace=0.30)
ax = np.array([fig.add_subplot(i) for i in gs])
for i, band in enumerate(fc.keys()):
    new_sc = sc[band]+w_degree2_flatten
    pval1,r1 = linear_fit(np.log10(new_sc), np.log10(fc[band]))
    ax[i].plot(np.log10(new_sc), np.log10(fc[band]), '.', color='navy', markersize=1)
    ax[i].plot(np.arange(-6,1), np.polyval(pval1, np.arange(-6,1)), 'orange', label='Fit all pairs')
    ax[i].set_title(f'{band:s}\nr = {r1:6.3f}, k = {pval1[0]:6.3f}', fontsize=19)
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].set_visible(True)
ax[-1].legend(handles, labels, loc=2, fontsize=20)
ax[-1].axis('off')
[fig.get_axes()[i].set_ylabel(r'$\log_{10}$(TDMI)') for i in (0,5)]
[fig.get_axes()[i].set_xlabel('$\log_{10}$(SC)') for i in (5,6,7,8)]
plt.savefig('Direct_Indirect_sc_fc_fitting_mi.png')

# %%
data = EcogGC()
data.init_data()
sc, fc = data.get_sc_fc('ch')

# %%
fig = plt.figure(figsize=(30,14))
gs = fig.add_gridspec(nrows=2, ncols=5, 
                      left=0.06, right=0.98, top=0.92, bottom=0.10, 
                      wspace=0.24, hspace=0.30)
ax = np.array([fig.add_subplot(i) for i in gs])
for i, band in enumerate(fc.keys()):
    # new_sc = sc[band]+w_degree2_flatten
    pval1,r1 = linear_fit(np.log10(sc[band]), np.log10(fc[band]))
    ax[i].plot(np.log10(sc[band]), np.log10(fc[band]), '.b', markersize=1)
    ax[i].plot(np.arange(-6,1), np.polyval(pval1, np.arange(-6,1)), 'b', label='Fit direct pairs')
    ax[i].plot(np.log10(w_degree2_flatten[w_degree2_flatten>0]), np.log10(fc[band][w_degree2_flatten>0]), '.r', markersize=1)
    pval2,r2 = linear_fit(np.log10(w_degree2_flatten[w_degree2_flatten>0]), np.log10(fc[band][w_degree2_flatten>0]))
    ax[i].plot(np.arange(-6,1), np.polyval(pval2, np.arange(-6,1)), 'r', label='Fit indirect pairs')
    ax[i].set_title(f'r(blue) = {r1:6.3f}, r(red) = {r2:6.3f} \n k(blue) = {pval1[0]:6.3f}, k(red) = {pval2[0]:6.3f}', fontsize=19)
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].set_visible(True)
ax[-1].legend(handles, labels, loc=2, fontsize=20)
ax[-1].axis('off')
[fig.get_axes()[i].set_ylabel(r'$\log_{10}$(GC)') for i in (0,5)]
[fig.get_axes()[i].set_xlabel('$\log_{10}$(SC)') for i in (5,6,7,8)]
plt.savefig('Direct_Indirect_sc_fc_fitting2color_gc.png')
# %%
