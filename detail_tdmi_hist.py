# %%
import numpy as np
import matplotlib.pyplot as plt
from fcpy.tdmi import compute_tdmi_full, compute_snr_matrix
from fcpy.core import *

# %%
data = EcogTDMI()
with open('tdmi_snr_analysis/snr_th_kmean_tdmi.pkl', 'rb') as f:
    snr_th = pickle.load(f)
# snr_th['above_delta'] = 8
data.init_data(snr_th=snr_th)
# data.init_data('tdmi_snr_analysis/', 'snr_th_kmean_tdmi.pkl')
sc, fc = data.get_sc_fc('ch')
roi_mask = data.roi_mask.copy()
snr_mask = data.get_snr_mask(snr_th=snr_th)
# %%
# Construct second order indirect connectivity matrix
smat = np.zeros_like(roi_mask, dtype=float)
smat[roi_mask] = sc['raw']
smat_bin = smat != 0
w_degree2 = np.zeros_like(smat)
for i in range(roi_mask.shape[0]):
    for j in range(roi_mask.shape[1]):
        if smat_bin[i,j] == 0 and i != j:
            mask = smat_bin[:, j] * smat_bin[i, :]
            w_degree2[i,j] = np.sum(smat[mask,j]*smat[i,mask])
w_degree2_flatten = w_degree2[roi_mask]

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
# fig.savefig('Direct_Indirect_MI_distribution.png')
# %%
band = 'above_delta'
fc_range = np.array((fc[band].min(), fc[band].max()))
fig, ax = plt.subplots(1,1,figsize=(12,9))
bins = 40
(counts_0, edges_0) = np.histogram(np.log10(fc[band].flatten()), bins=bins, range=np.log10(fc_range))
(counts_1, edges_1) = np.histogram(np.log10(fc[band][w_degree2_flatten>0]), bins=bins, range=np.log10(fc_range))
(counts_2, edges_2) = np.histogram(np.log10(fc[band][sc[band]>0]), bins=bins, range=np.log10(fc_range))
(counts_3, edges_3) = np.histogram(np.log10(fc[band][w_degree2_flatten>1e-4]), bins=bins, range=np.log10(fc_range))
(counts_4, edges_4) = np.histogram(np.log10(fc[band][sc[band]>1e-2]), bins=bins, range=np.log10(fc_range))
ax.plot(edges_0[1:], counts_0, ds='steps-pre', color='k', label='All Pairs')
ax.plot(edges_1[1:], counts_1, ds='steps-pre', color='b', label='Indirect SC')
ax.plot(edges_2[1:], counts_2, ds='steps-pre', color='r', label='Direct SC')
ax.plot(edges_3[1:], counts_3, ds='steps-pre', color='orange', label='Strong indirect SC')
ax.plot(edges_4[1:], counts_4, ds='steps-pre', color='g', label='Strong direct SC')
plt.legend()
ax.set_title(band)
ax.set_xlabel(r'$\log_{10}$(TDMI)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
ax.set_ylim(0)
# %%
direct_sc = sc['raw'].copy()
direct_sc[direct_sc==0] = np.nan
direct_sc[direct_sc==1.5] = np.nan
new_sc = sc['raw']+w_degree2_flatten
new_sc[new_sc==1.5] = np.nan

fc_range = np.array((np.nanmin(new_sc), np.nanmax(new_sc)))
fig, ax = plt.subplots(1,1,figsize=(12,9))
bins = 50

for data2plot, label in zip(
    (
        # new_sc,
        direct_sc,
        w_degree2_flatten[w_degree2_flatten>0],
    ),
    (
        # 'All Pairs',
        'Direct SC',
        'Indirect SC',
    ),

):
    (counts, edges) = np.histogram(np.log10(data2plot), bins=bins, range=np.log10(fc_range))
    # ax.plot(edges[1:], counts, ds='steps-pre', label=label, fillstyle='bottom')
    ax.bar(edges[1:], counts, width=edges[1]-edges[0], label=label, alpha=.5)
ax.legend()
ax.set_xlabel(r'$\log_{10}$(SC)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
ax.set_ylim(0)
ax.legend()
fig.savefig('sc_all_hist.png')
# %%
band = 'above_delta'
tdmi_data = compute_tdmi_full(data.tdmi_data[band])[roi_mask, :]
snr_metric = compute_snr_matrix(data.tdmi_data[band])[roi_mask]

# %%
fc_pick_range = np.array([10**-1.8, 10**-1.6])
fc_pick_mask = (fc[band]>fc_pick_range[0])*(fc[band]<fc_pick_range[1])
fc_pick_nonzero = np.nonzero(fc_pick_mask)[0]
# %%
idx = 1
idxs = np.random.choice(np.arange(fc_pick_nonzero.shape[0]), 10, replace=False)
for idx in idxs:
    fig = plt.figure()
    plt.plot(np.arange(-3000,3001), tdmi_data[fc_pick_nonzero[idx]])
    argmax = np.argmax(tdmi_data[fc_pick_nonzero[idx]])
    plt.axvline(np.arange(-3000,3001)[argmax], ls='--', color='orange', label=f'delay = {np.arange(-3000,3001)[argmax]:d}ms')
    plt.title(f"direct sc = {sc[band][fc_pick_nonzero[idx]]:5.2e}\n"
        + f"indirect sc = {w_degree2_flatten[fc_pick_nonzero[idx]]:5.2e}\n"
        + f"snr value = {snr_metric[fc_pick_nonzero[idx]]:4.1f}"
    )
    # plt.xlim(-500,500)
    plt.legend()

# %%
from fcpy.plot_frame import *
from fcpy.plot import gen_fc_rank_figure_single
data_plt = {}
for band in data.filters:
    data_plt[band] = {
        'fc':fc[band],
        'sc':sc[band],
        'band':band,
        'snr_mask':snr_mask[band],
    }

fig = fig_frame52(data_plt, gen_fc_rank_figure_single)
ax = fig.get_axes()
[axi.set_ylabel('Ranked TDMI index') for axi in ax if axi.get_ylabel()]
[axi.set_xlabel(r'$\log_{10}$(TDMI value)') for axi in ax if axi.get_xlabel()]
# %%
band = 'above_delta'
fc_range = np.array((fc[band].min(), fc[band].max()))
fig, ax = plt.subplots(1,1,figsize=(12,9))
bins = 40
(counts_0, edges_0) = np.histogram(np.log10(fc[band].flatten()), bins=bins, range=np.log10(fc_range))
(counts_1, edges_1) = np.histogram(np.log10(fc[band][w_degree2_flatten>0]), bins=bins, range=np.log10(fc_range))
(counts_2, edges_2) = np.histogram(np.log10(fc[band][sc[band]>0]), bins=bins, range=np.log10(fc_range))
(counts_3, edges_3) = np.histogram(np.log10(fc[band][snr_mask[band]]), bins=bins, range=np.log10(fc_range))
(counts_4, edges_4) = np.histogram(np.log10(fc[band][sc[band]>1e-2]), bins=bins, range=np.log10(fc_range))
ax.plot(edges_0[1:], counts_0, ds='steps-pre', color='k', label='All Pairs')
ax.plot(edges_1[1:], counts_1, ds='steps-pre', color='b', label='Indirect SC')
ax.plot(edges_2[1:], counts_2, ds='steps-pre', color='r', label='Direct SC')
ax.plot(edges_3[1:], counts_3, ds='steps-pre', color='orange', label='Strong SNR Pairs')
ax.plot(edges_4[1:], counts_4, ds='steps-pre', color='g', label='Strong direct SC')
plt.legend()
ax.set_title(band)
ax.set_xlabel(r'$\log_{10}$(TDMI)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
ax.set_ylim(0)
# %%
band = 'above_delta'
fc_range = np.array((fc[band].min(), fc[band].max()))
fig, ax = plt.subplots(1,1,figsize=(12,9))
bins = 40

for data2plot, label in zip(
    (
        fc[band].flatten(),
        fc[band][w_degree2_flatten>0],
        fc[band][sc[band]>0],
        fc[band][sc[band]>1e-2],
        fc[band][sc[band]<=1e-2],
    ),
    (
        'All Pairs',
        'Indirect SC',
        'Direct SC',
        'Strong direct SC',
        'Weak direct SC',
    ),

):
    (counts, edges) = np.histogram(np.log10(data2plot), bins=bins, range=np.log10(fc_range))
    ax.plot(edges[1:], counts, ds='steps-pre', label=label)
ax.legend(fontsize=16)
ax.set_title(band)
ax.set_xlabel(r'$\log_{10}$(TDMI)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
ax.set_ylim(0)
fig.savefig('fc_direct_sc_hist.png')
# %%
band = 'above_delta'
fc_range = np.array((1e-6, 1))
fig, ax = plt.subplots(1,1,figsize=(12,9))
bins = 40

for data2plot, label, color, linestyle in zip(
    (
        new_sc.flatten(),
        w_degree2_flatten,
        w_degree2_flatten[fc[band]>10**-1.5],
        sc[band],
        sc[band][fc[band]>10**-1.5],
    ),
    (
        'All Pairs',
        'Indirect SC',
        'Indirect SC (Large TDMI)',
        'Direct SC',
        'Direct SC (Large TDMI)',
    ),
    (
        'k',
        'r',
        'r',
        'g',
        'g',
    ),
    (
        '-',
        '-',
        '--',
        '-',
        '--',
    ),

):
    (counts, edges) = np.histogram(np.log10(data2plot), bins=bins, range=np.log10(fc_range))
    ax.plot(edges[1:], counts, ds='steps-pre', label=label, color=color, ls = linestyle)
ax.legend(fontsize=16)
ax.set_title(band)
ax.set_xlabel(r'$\log_{10}$(SC)', fontsize=16)
ax.set_ylabel('Counts', fontsize=16)
ax.set_ylim(0)
fig.savefig('sc_hist_classified.png')

# %%
