"""
======================
Random Geometric Graph
======================

Example
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle

path = 'tdmi_snr_analysis/'
data_package = np.load(path+'preprocessed_data.npz', allow_pickle=True)
with open(path + 'WA_v3_cg.pkl', 'rb') as f:
    tdmi_mask_total = pickle.load(f)

weight = data_package['adj_mat']
FROM, TO = np.meshgrid(range(weight.shape[0]), range(weight.shape[1]))

edges = [(i,j,w) for i,j,w in zip(FROM.flatten(), TO.flatten(), weight.flatten())]

n = 46
G = nx.DiGraph()
G.add_nodes_from(range(n))
G.add_weighted_edges_from(edges)
# position is stored as node attribute data for random_geometric_graph
np.random.seed(0)
pos_x, pos_y = np.meshgrid(np.arange(7)/7, np.arange(7)/7)
pos_x, pos_y = pos_x.flatten()+.1, pos_y.flatten()+.1
indices =np.random.choice(np.arange(pos_x.shape[0]), n, replace=False).astype(int)
indices = np.arange(n).astype(int)
pos = {i: [pos_x[indices[i]], pos_y[indices[i]]] for i in range(n)}
labels = {n:str(n) for n,_ in pos.items()}

# find node near center (0.5,0.5)
dmin = 1
ncenter = 0
# for n in pos:
#     x, y = pos[n]
#     d = (x - 0.5) ** 2 + (y - 0.5) ** 2
#     if d < dmin:
#         ncenter = n
#         dmin = d

# color by path length from node near center
# p = dict(nx.single_source_shortest_path_length(G, ncenter))

filter_pool = ['delta', 'theta', 'alpha',
                'beta', 'gamma', 'high_gamma']
color_dict = {
    'delta':      (1,  0,  0,  1),
    'theta':      (1, .5,  0,  1),
    'alpha':      (1,  1,  0,  1),
    'beta':       (0,  1,  0,  1),
    'gamma':      (0,  1,  1,  1),
    'high_gamma': (1,  0,  1,  1)
}

# color_array = np.zeros((len(edges), 4))
# for band in filter_pool:
#     buffer = tdmi_mask_total["## $w_{ij}>10^{-3}$ "][band]
#     color_array[buffer.flatten(), :] = color_dict[band]

title_set = [
    "## $w_{ij}>10^{%d}$ " % item for item in np.arange(-6, 1)
]
for title in title_set:
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    for i in range(6):
        color_array = np.zeros((len(edges), 4))
        buffer = tdmi_mask_total[title][filter_pool[i]]
        color_array[buffer.flatten(), :] = color_dict[filter_pool[i]]
        nx.draw_networkx_edges(
            G, 
            pos, 
            # nodelist=[ncenter], 
            edge_color=color_array,
            ax = ax[i],
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            # nodelist=list(p.keys()),
            node_size=400,
            node_color='w',
            edgecolors='navy',
            # node_color=list(p.values()),
            cmap=plt.cm.Reds_r,
            ax = ax[i],
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels={n:lab for n, lab in labels.items() if n in pos},
            font_size=10,
            ax = ax[i],
        )

        ax[i].set_xlim(-0.05, 1.05)
        ax[i].set_ylim(-0.05, 1.05)
        ax[i].set_title(filter_pool[i], fontsize=20)
        ax[i].axis("off")
    fig.suptitle(title.strip('#'), fontsize=20)
    plt.tight_layout()
    plt.savefig(path + f"network_graph_{title.strip('#$ '):s}.png")
