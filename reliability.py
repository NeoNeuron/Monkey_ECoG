#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# %%
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size']=20
from fcpy.core import EcogTDMI, EcogCC
from fcpy.utils import linear_fit
# %%

path = 'data/'
dfname1 = 'tdmi_data_part1_3k.npz'
dfname2 = 'tdmi_data_part2_3k.npz'
# dfname1 = 'cc_part1.npz'
# dfname2 = 'cc_part2.npz'

tdmi_data_1 = EcogTDMI(path, dfname1)
tdmi_data_2 = EcogTDMI(path, dfname2)
# tdmi_data_1.init_data()
# tdmi_data_2.init_data()
tdmi_data_1.init_data('tdmi_snr_analysis/')
tdmi_data_2.init_data('tdmi_snr_analysis/')
sc, fc1 = tdmi_data_1.get_sc_fc('ch')
sc, fc2 = tdmi_data_2.get_sc_fc('ch')
# %%
fig = plt.figure(figsize=(25,14))
gs = fig.add_gridspec(nrows=2, ncols=4, 
                      left=0.06, right=0.98, top=0.92, bottom=0.10, 
                      wspace=0.24, hspace=0.30)
ax = np.array([fig.add_subplot(i) for i in gs])
for i, band in enumerate(sc.keys()):
    ax[i].plot(np.log10(fc1[band]), np.log10(fc2[band]), '.k')
    ax[i].axis('scaled')
    ax[i].grid(ls='--', color='grey', lw=0.5)
    pval, r = linear_fit(np.log10(fc1[band]), np.log10(fc2[band]))
    # ax[i].plot(np.arange(-2, 1), np.arange(-2, 1), '--', color='orange')
    # ax[i].plot(np.arange(-2, 1), np.polyval(pval, np.arange(-2,1)), label='fit')
    ax[i].set_title(f'r = {r:6.3f}')
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].set_visible(True)
ax[-1].legend(handles, labels, loc=2, fontsize=20)
ax[-1].axis('off')
[ax[i].set_xlabel('MI value in trial 1') for i in (3,4,5,6)]
[ax[i].set_ylabel('MI value in trial 2') for i in (0,4)]

# %%
plt.savefig(f'reliability_mi.png')
