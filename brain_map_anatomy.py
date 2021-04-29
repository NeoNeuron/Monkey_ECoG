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
weight = data_package['weight'].copy()

with open(path + 'tdmi_kmean/recon_gap_tdmi.pkl', 'rb') as f:
    sc_mask = pickle.load(f)
    _ = pickle.load(f)
    roi_mask = pickle.load(f)

for idx in range(len(sc_mask)):
    buffer = np.zeros_like(roi_mask, dtype=bool)
    buffer[roi_mask] = sc_mask[idx]
    buffer[weight==1.5] = False
    sc_mask[idx] = buffer.copy()  # hide intra-area connections


# position is stored as node attribute data for random_geometric_graph
n = weight.shape[0]
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
for i, mask in enumerate(sc_mask):
    ax[i].plot(loc[:, 0], loc[:, 1], 'or', ms=3)
    ax[i].axis('equal')
    if i < len(sc_mask) - 1:
        mask_buffer = mask*(~sc_mask[i+1])
    else:
        mask_buffer = mask.copy()
    alpha_val = 1./np.sqrt(np.sum(mask_buffer))+0.005
    for x, y, c in zip(xx.flatten(), yy.flatten(), mask_buffer.flatten()):
        if c:
            ax[i].plot(loc[(x,y),0], loc[(x,y),1], color='k', alpha=alpha_val)
    ax[i].set_title(r'$10^{%.d}$<w<$10^{%.d}$'%(i-6, i-5), fontsize=20)
    ax[i].invert_yaxis()
    ax[i].axis("off")

norm = matplotlib.colors.Normalize(vmin=-6, vmax=np.log10(1.5))
ax[-1].axis("off")

plt.savefig(path + f"anatomy_band.png")


fig = plt.figure(figsize=(9,15), dpi=100)
gs = fig.add_gridspec(nrows=4, ncols=2, 
                      left=0.05, right=0.96, top=0.96, bottom=0.05, 
                      wspace=0.15, hspace=0.20)
ax = np.array([fig.add_subplot(i) for i in gs])
xx, yy = np.meshgrid(np.arange(117), np.arange(117))
w_max=weight.max()
alphas = [0.007, 0.007, 0.01, 0.02, 0.05, 0.1, 0.15]
for i, mask in enumerate(sc_mask):
    ax[i].plot(loc[:, 0], loc[:, 1], 'or', ms=3)
    ax[i].axis('equal')
    # alpha_val = 1./np.sqrt(np.sum(mask))+0.005
    for x, y, c in zip(xx.flatten(), yy.flatten(), mask.flatten()):
        if c:
            ax[i].plot(loc[(x,y),0], loc[(x,y),1], color='k', alpha=alphas[i])
    ax[i].set_title(r'$w>10^{-%.d}$'%(6-i), fontsize=20)
    ax[i].invert_yaxis()
    ax[i].axis("off")

ax[-1].invert_yaxis()
ax[-1].axis("off")
# norm = matplotlib.colors.Normalize(vmin=-6, vmax=np.log10(1.5))
# ax[-1].plot(loc[:, 0], loc[:, 1], 'ok', ms=3)
# for x, y, c in zip(xx.flatten(), yy.flatten(), weight.flatten()):
#     ax[-1].plot(loc[(x,y),0], loc[(x,y),1], color=rb_cmap(norm(np.log10(c+1e-6))), alpha=0.01+0.99*c/w_max)
# ax[-1].axis('equal')
# ax[-1].axis("off")
# ax[-1].set_title('Anatomical Weight', fontsize=20)

plt.savefig(path + f"recon_answer.png")