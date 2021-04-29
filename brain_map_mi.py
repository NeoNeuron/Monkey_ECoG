"""
======================
Random Geometric Graph
======================

Example
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

path = 'tdmi_snr_analysis/'
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)

with open(path + 'tdmi_kmean/recon_gap_tdmi.pkl', 'rb') as f:
    sc_mask = pickle.load(f)
    fc_mask = pickle.load(f)
    roi_mask = pickle.load(f)

for band, item in fc_mask.items():
    buffer = np.zeros_like(roi_mask, dtype=bool)
    buffer[roi_mask] = item
    fc_mask[band] = buffer.copy()

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
for i, band in enumerate(color_dict.keys()):
    ax[i].plot(loc[:, 0], loc[:, 1], 'ok', ms=3)
    ax[i].axis('equal')
    for x, y, c in zip(xx.flatten(), yy.flatten(), fc_mask[band].flatten()):
        if c:
            ax[i].plot(loc[(x,y),0], loc[(x,y),1], color=color_dict[band], alpha=0.05)
    ax[i].set_title(band, fontsize=20)
    ax[i].invert_yaxis()
    ax[i].axis("off")

ax[-1].axis("off")

plt.savefig(path + f"recon_net_graph.png")