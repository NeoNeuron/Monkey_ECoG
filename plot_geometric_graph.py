"""
======================
Random Geometric Graph
======================

Example
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
with open(path + 'tdmi_kmean/recon_gap_tdmi.pkl', 'rb') as f:
    sc_mask = pickle.load(f)
    fc_mask = pickle.load(f)
    roi_mask = pickle.load(f)

for idx in range(len(sc_mask)):
    buffer = np.zeros_like(roi_mask, dtype=bool)
    buffer[roi_mask] = sc_mask[idx]
    sc_mask[idx] = buffer.copy()

for band, item in fc_mask.items():
    buffer = np.zeros_like(roi_mask, dtype=bool)
    buffer[roi_mask] = item
    fc_mask[band] = buffer.copy()

weight = data_package['weight']
FROM, TO = np.meshgrid(range(weight.shape[0]), range(weight.shape[1]))

edges = [(i,j,w) for i,j,w in zip(FROM.flatten(), TO.flatten(), weight.flatten())]

# position is stored as node attribute data for random_geometric_graph
n = 117
loc = data_package['loc']

# color by path length from node near center
rb_cmap = plt.cm.get_cmap('rainbow_r')
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
color_dict = {band: rb_cmap(idx/6.0) for idx, band in enumerate(filter_pool)}
color_dict['raw'] = 'k'

fig = plt.figure(figsize=(9,15), dpi=100)
gs = fig.add_gridspec(nrows=4, ncols=2, 
                      left=0.05, right=0.96, top=0.96, bottom=0.05, 
                      wspace=0.15, hspace=0.20)
ax = np.array([fig.add_subplot(i) for i in gs])
xx, yy = np.meshgrid(np.arange(117), np.arange(117))
w_max=weight.max()
for i, band in enumerate(color_dict.keys()):
    ax[i].plot(loc[:, 0], loc[:, 1], 'ok', ms=3)
    ax[i].axis('equal')
    for x, y, c in zip(xx.flatten(), yy.flatten(), fc_mask[band].flatten()):
        if c:
            ax[i].plot(loc[(x,y),0], loc[(x,y),1], color=color_dict[band], alpha=0.05)
    ax[i].set_title(band, fontsize=20)
    ax[i].axis("off")

norm = matplotlib.colors.Normalize(vmin=-6, vmax=np.log10(1.5))
ax[-1].plot(loc[:, 0], loc[:, 1], 'ok', ms=3)
for x, y, c in zip(xx.flatten(), yy.flatten(), weight.flatten()):
    ax[-1].plot(loc[(x,y),0], loc[(x,y),1], color=rb_cmap(norm(np.log10(c+1e-6))), alpha=0.01+0.99*c/w_max)
ax[-1].axis('equal')
ax[-1].axis("off")
ax[-1].set_title('Anatomical Weight', fontsize=20)

plt.savefig(path + f"recon_net_graph.png")


fig = plt.figure(figsize=(9,15), dpi=100)
gs = fig.add_gridspec(nrows=4, ncols=2, 
                      left=0.05, right=0.96, top=0.96, bottom=0.05, 
                      wspace=0.15, hspace=0.20)
ax = np.array([fig.add_subplot(i) for i in gs])
xx, yy = np.meshgrid(np.arange(117), np.arange(117))
w_max=weight.max()
alphas = [0.006, 0.006, 0.01, 0.02, 0.05, 0.1, 0.15]
for i, mask in enumerate(sc_mask):
    ax[i].plot(loc[:, 0], loc[:, 1], 'or', ms=3)
    ax[i].axis('equal')
    for x, y, c in zip(xx.flatten(), yy.flatten(), mask.flatten()):
        if c:
            ax[i].plot(loc[(x,y),0], loc[(x,y),1], color='k', alpha=alphas[i])
    ax[i].set_title(r'$w>10^{-%.d}$'%(6-i), fontsize=20)
    ax[i].axis("off")

ax[-1].axis("off")
# norm = matplotlib.colors.Normalize(vmin=-6, vmax=np.log10(1.5))
# ax[-1].plot(loc[:, 0], loc[:, 1], 'ok', ms=3)
# for x, y, c in zip(xx.flatten(), yy.flatten(), weight.flatten()):
#     ax[-1].plot(loc[(x,y),0], loc[(x,y),1], color=rb_cmap(norm(np.log10(c+1e-6))), alpha=0.01+0.99*c/w_max)
# ax[-1].axis('equal')
# ax[-1].axis("off")
# ax[-1].set_title('Anatomical Weight', fontsize=20)

plt.savefig(path + f"recon_answer.png")