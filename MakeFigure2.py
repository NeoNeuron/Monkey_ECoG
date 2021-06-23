#! /usr/bin/python 
# Author: Kai Chen

# Make the Figure 2 for paper: Relationship between 1/distance and FC(TDMI).

# %%
from utils.core import *
import numpy as np
import matplotlib.pyplot as plt
from MakeFigure1 import gen_sc_fc_figure_new, axis_log_formater, spines_formater

@spines_formater
@axis_log_formater(axis='y')
def gen_sc_fc_figure(*args, **kwargs):
    return gen_sc_fc_figure_new.__wrapped__.__wrapped__(*args, **kwargs)
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

fig, ax = plt.subplots(1,2,figsize=(9,4), dpi=400)

band = 'raw'
# new_mask = np.ones_like(snr_mask_tdmi[band])
new_mask = snr_mask_tdmi[band].copy()
# new_mask[sc_tdmi[band]==0] = False
# new_mask[sc_tdmi[band]==1.5] = False
gen_sc_fc_figure(ax[0], fc_tdmi[band], 1./d_mat[band], new_mask, is_log='y')
gen_sc_fc_figure(ax[1], fc_gc[band],   1./d_mat[band], new_mask, is_log='y')

for axi, labeli in zip(ax, ('TDMI', 'GC')):
    axi.set_title(axi.get_title().replace(band, labeli))
    axi.set_xlabel(r'$1/Distance$')
    axi.set_ylabel(labeli)
fig.suptitle(band)

fig.savefig(path+'Figure_2.png')
# %%
