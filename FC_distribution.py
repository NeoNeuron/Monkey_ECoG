# %%
import numpy as np
import matplotlib.pyplot as plt
from fcpy.core import EcogTDMI, EcogGC, EcogCC
from fcpy.plot_frame import plot_union

# %%
data = EcogTDMI()
data.init_data('tdmi_snr_analysis/')
# data.init_data()
sc, fc = data.get_sc_fc('ch')

data_plt = {}
for band in fc.keys():
    data_plt[band] = {'sc': sc[band], 'fc': fc[band]}

# %%
def plot_fc_distribution(ax, data:dict):
    (counts_0, edges_0) = np.histogram(np.log10(data['fc'][data['sc']==0]), bins=100)
    (counts_1, edges_1) = np.histogram(np.log10(data['fc'][data['sc'] >0]), bins=100)
    ax.plot(edges_0[1:], counts_0, ds='steps-pre', color='navy', label='SC Absent')
    ax.plot(edges_1[1:], counts_1, ds='steps-pre', color='orange', label='SC Present')
    ax.set_xlabel(r'$\log_{10}$(TDMI)')
    ax.set_ylabel('Counts')
    return ax

fig = plot_union(data_plt, plot_fc_distribution)
plt.savefig('TDMI_distribution.png')
plt.close()
# %%
data = EcogCC()
data.init_data()
sc, fc = data.get_sc_fc('ch')
cc_shuffle = np.load('data/cc_shuffled.npz', allow_pickle=True)

data_plt = {}
for band in fc.keys():
    data_plt[band] = {'sc': sc[band], 'fc': fc[band], 'shuffle': cc_shuffle[band]}

def plot_cc_distribution(ax, data:dict):
    (counts_0, edges_0) = np.histogram((data['fc'][data['sc']==0]), bins=100)
    (counts_1, edges_1) = np.histogram((data['fc'][data['sc'] >0]), bins=100)
    ax.plot(edges_0[1:], counts_0, ds='steps-pre', color='navy', label='SC Absent')
    ax.plot(edges_1[1:], counts_1, ds='steps-pre', color='orange', label='SC Present')
    ax.axvline(data['shuffle'].std(), color='r', alpha=.5, label='Base Line')
    ax.set_xlabel(r'$\log_{10}$(FC)')
    ax.set_ylabel('Counts')
    ax.set_xlim(0,1)
    return ax

# plot legend in the empty subplot
fig = plot_union(data_plt, plot_cc_distribution)
fig.savefig('CC_distribution.png')
plt.close()
