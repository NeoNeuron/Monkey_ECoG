#! /usr/bin/python 
# Author: Kai Chen

# Make the Figure 5 for paper: Shortest Path Distance Anaylsis
# * Figure 5-1   : Shortest path distance v.s. TDMI
# * Figure 5-2   : Direct distance v.s. Shortest path distance v.s. diff(TDMI-SC)
# * Figure 5-3   : Sorted colormap for diff(TDMI-SC)
# * Figure 5-3-1 : SC v.s. TDMI masked by diff(TDMI-SC)
# * Figure 5-4   : Direct distance v.s. Shortest path distance v.s. diff(GC-SC)
# * Figure 5-5   : Sorted colormap for diff(GC-SC)

# %%
from fcpy.core import *
from MakeFigure1 import axis_log_formater, gen_sc_fc_figure_new, spines_formater
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %%
path = 'image/'
data_tdmi = EcogTDMI()
data_tdmi.init_data(path, 'snr_th_kmean_tdmi.pkl')
sc_tdmi, fc_tdmi = data_tdmi.get_sc_fc('ch')
snr_mask_tdmi = data_tdmi.get_snr_mask(path, 'snr_th_kmean_tdmi.pkl')
roi_mask = data_tdmi.roi_mask.copy()

data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
dist_mat = data_package['d_matrix']
d_mat = {band: dist_mat[roi_mask] for band in data_tdmi.filters}

data_gc = EcogGC()
data_gc.init_data()
sc_gc, fc_gc = data_gc.get_sc_fc('ch')
# %%
# Compute the shortest path length
band = 'raw'
con_2d = np.zeros_like(roi_mask, dtype=float)
con_2d[roi_mask] = sc_tdmi[band]

G = nx.DiGraph(1./con_2d)
shortest_path_length = dict(nx.all_pairs_dijkstra_path_length(G))
shortest_path_length_array = np.zeros_like(con_2d, dtype=float)
for i in range(shortest_path_length_array.shape[0]):
    for j in range(shortest_path_length_array.shape[1]):
        shortest_path_length_array[i,j] = shortest_path_length[i][j]
# %%
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=400)
band = 'raw'
# new_mask = np.ones_like(snr_mask_tdmi[band])
new_mask = snr_mask_tdmi[band].copy()
new_mask[sc_tdmi[band]==0] = False
new_mask[sc_tdmi[band]==1.5] = False
gen_sc_fc_figure_new(ax[0], fc_tdmi[band], 1./shortest_path_length_array[roi_mask], new_mask,)
gen_sc_fc_figure_new(ax[1], fc_gc[band], 1./shortest_path_length_array[roi_mask], new_mask,)

for axi, labeli in zip(ax, ('TDMI', 'GC')):
    axi.set_title(axi.get_title().replace(band, labeli))
    axi.set_xlabel('1/Shortest Path Length')
    axi.set_ylabel(labeli)
fig.suptitle(band)
fig.savefig(path+'Figure_5-1.png')

# %%
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=400)
band = 'raw'
fc_zscore = np.log10(fc_tdmi[band])
sc_zscore = np.log10(sc_tdmi[band]+1e-6)
normalize = lambda x: (x-x.mean())/x.std()
diff = normalize(fc_zscore)-normalize(sc_zscore)
vmax = np.abs(diff).max()

@spines_formater
@axis_log_formater(axis='y')
@axis_log_formater(axis='x')
def gen_sp_p(ax):
    cax = ax.scatter(np.log10(1./con_2d[roi_mask])[new_mask], np.log10(shortest_path_length_array[roi_mask])[new_mask], 
        c=diff[new_mask], vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r,
        alpha=1, s=5,zorder=10, 
)
    plt.colorbar(cax, ax=ax)
    ax.set_xlabel('Direct distance')
    ax.set_ylabel('Shortest path distance')
    ax.plot(np.linspace(0,2,10),np.linspace(0,2,10), 'k')
    return ax
gen_sp_p(ax[0])

@axis_log_formater(axis='x')
@spines_formater
def gen_sc_fc_figure(*args, **kwargs):
    return gen_sc_fc_figure_new.__wrapped__.__wrapped__(*args, **kwargs)
gen_sc_fc_figure(ax[1], diff, 1./(con_2d[roi_mask]*shortest_path_length_array[roi_mask]), new_mask, c=diff, is_log='x')
ax[1].set_xlabel('Indirect path factor')
ax[1].set_ylabel('Diff. (FC-SC)')

plt.tight_layout()
fig.savefig(path+'Figure_5-2.png')


# %%
diff_mat = np.ones_like(roi_mask, dtype=float)
diff_mat[roi_mask] = diff
# reorder diff_mat
diff_mat_mean = diff_mat.sum(0) + diff_mat.sum(1)
diff_order = np.flip(np.argsort(diff_mat_mean))
diff_mat_reorder = diff_mat.copy()
diff_mat_reorder = diff_mat_reorder[diff_order, :]
diff_mat_reorder = diff_mat_reorder[:, diff_order]

fig, ax = plt.subplots(1,3,dpi=400, figsize=(14,4))

vmax = np.abs(diff).max()
plt.cm.RdBu_r(diff)
cax=ax[0].scatter(normalize(sc_zscore), normalize(fc_zscore), c=diff, vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r)
plt.colorbar(cax, ax=ax[0])
ax[0].set_xlabel('SC Score')
ax[0].set_ylabel('TDMI Score')
ax[0].plot(np.linspace(-1,2,10),np.linspace(-1,2,10), 'k')

cax = ax[1].pcolormesh(diff_mat_reorder, vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r)
plt.colorbar(cax, ax=ax[1])
ax[1].axis('scaled')
ax[1].invert_yaxis()

# position is stored as node attribute data for random_geometric_graph
n = diff_mat.shape[0]
loc = data_package['loc']

# color by path length from node near center
xx, yy = np.meshgrid(np.arange(117), np.arange(117))
ax[2].scatter(loc[:, 0], loc[:, 1], s=40, lw=0, c=diff_order, cmap=plt.cm.RdBu_r)
ax[2].plot((100, 100), (1050,1150), 'k')
ax[2].plot((100, 200), (1150,1150), 'k')
ax[2].text(0.1, 0.1, '100', transform=ax[2].transAxes)
ax[2].axis('equal')
ax[2].invert_yaxis()
ax[2].axis("off")

fig.savefig(path+'Figure_5-3.png')
# %%
shortest_path_mask = 1./con_2d - shortest_path_length_array 
shortest_path_mask = (shortest_path_mask[roi_mask] < 10)
fig, ax = plt.subplots(1,2,figsize=(9,4), dpi=400)

band = 'raw'
# new_mask = np.ones_like(snr_mask_tdmi[band])
new_mask = snr_mask_tdmi[band].copy() * shortest_path_mask.copy()
new_mask[sc_tdmi[band]==0] = False
new_mask[sc_tdmi[band]==1.5] = False
gen_sc_fc_figure_new(ax[0], fc_tdmi[band], sc_tdmi[band], new_mask,)
gen_sc_fc_figure_new(ax[1], fc_gc[band], sc_gc[band], new_mask,)

for axi, labeli in zip(ax, ('TDMI', 'GC')):
    axi.set_title(labeli+' : '+axi.get_title())
    # axi.set_xlim(-5.5, 0)
    axi.set_ylabel(labeli)
fig.suptitle(band)

fig.savefig(path+'Figure_5-3-1.png')

# %%
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=400)
band = 'raw'
new_mask = snr_mask_tdmi[band].copy()
new_mask *= (sc_tdmi[band]!=1.5)
fc_zscore = np.log10(fc_gc[band])
sc_zscore = np.log10(sc_gc[band]+1e-6)
normalize = lambda x: (x-x.mean())/x.std()
diff = normalize(fc_zscore)-normalize(sc_zscore)

@spines_formater
@axis_log_formater(axis='y')
@axis_log_formater(axis='x')
def gen_sp_p(ax):
    cax = ax.scatter(np.log10(1./con_2d[roi_mask])[new_mask], np.log10(shortest_path_length_array[roi_mask])[new_mask], 
        c=diff[new_mask], vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r,
        s=5, alpha=1,zorder=10,
    )
    plt.colorbar(cax, ax=ax)
    ax.set_xlabel('Direct distance')
    ax.set_ylabel('Shortest path distance')
    ax.plot(np.linspace(0,2,10),np.linspace(0,2,10), 'k')
    return ax
gen_sp_p(ax[0])

@axis_log_formater(axis='x')
@spines_formater
def gen_sc_fc_figure(*args, **kwargs):
    return gen_sc_fc_figure_new.__wrapped__.__wrapped__(*args, **kwargs)
gen_sc_fc_figure(ax[1], diff[new_mask], 1./(con_2d[roi_mask]*shortest_path_length_array[roi_mask])[new_mask], c=diff[new_mask], is_log='x')
ax[1].set_xlabel('Indirect path factor')
ax[1].set_ylabel('Diff. (FC-SC)')

plt.tight_layout()
fig.savefig(path+'Figure_5-4.png')
# %%
diff_mat = np.ones_like(roi_mask, dtype=float)
diff_mat[roi_mask] = diff
# reorder diff_mat
diff_mat_mean = diff_mat.sum(0) + diff_mat.sum(1)
diff_order = np.flip(np.argsort(diff_mat_mean))
diff_mat_reorder = diff_mat.copy()
diff_mat_reorder = diff_mat_reorder[diff_order, :]
diff_mat_reorder = diff_mat_reorder[:, diff_order]

fig, ax = plt.subplots(1,3,dpi=300, figsize=(14,4))
vmax = np.abs(diff).max()
plt.cm.RdBu_r(diff)
cax=ax[0].scatter(normalize(sc_zscore), normalize(fc_zscore), c=diff, vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r)
plt.colorbar(cax, ax=ax[0])
ax[0].set_xlabel('SC Score')
ax[0].set_ylabel('GC Score')
ax[0].plot(np.linspace(-1,2,10),np.linspace(-1,2,10), 'k')

cax = ax[1].pcolormesh(diff_mat_reorder, vmax=vmax, vmin=-vmax, cmap=plt.cm.RdBu_r)
plt.colorbar(cax, ax=ax[1])
ax[1].axis('scaled')
ax[1].invert_yaxis()


# position is stored as node attribute data for random_geometric_graph
n = diff_mat.shape[0]
loc = data_package['loc']

# color by path length from node near center
xx, yy = np.meshgrid(np.arange(117), np.arange(117))
ax[2].scatter(loc[:, 0], loc[:, 1], s=40, lw=0, c=diff_order, cmap=plt.cm.RdBu_r)
ax[2].plot((100, 100), (1050,1150), 'k')
ax[2].plot((100, 200), (1150,1150), 'k')
ax[2].text(0.1, 0.1, '100', transform=ax[2].transAxes)
ax[2].axis('equal')
ax[2].invert_yaxis()

ax[2].axis("off")

fig.savefig(path+'Figure_5-5.png')
# %%
