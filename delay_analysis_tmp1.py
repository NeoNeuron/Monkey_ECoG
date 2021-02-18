import os
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from draw_causal_distribution_v2 import load_data
from tdmi_delay_analysis_v1 import get_delay_matrix


path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
band = 'raw'
tdmi_data = load_data(path, band)
n_channel = tdmi_data.shape[0]
n_delay = tdmi_data.shape[2]

# complete the tdmi series
# tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
# tdmi_data_full[:,:,n_delay-1:] = tdmi_data
# tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)

delay_mat = get_delay_matrix(path, band, force_compute=True)
fig, ax = plt.subplots(4,1, figsize=(4,12), sharex=True)
ax[0].pcolormesh(delay_mat==0, cmap=plt.cm.gray)

ax[1].pcolormesh(weight==1.5, cmap=plt.cm.gray)

ax[2].pcolormesh((delay_mat==0)*(weight==1.5), cmap=plt.cm.gray)

ax[3].pcolormesh((delay_mat==0)*~(weight==1.5), cmap=plt.cm.gray)

[axi.invert_yaxis() for axi in ax]
[axi.axis('scaled') for axi in ax]

plt.tight_layout()
plt.savefig('tmp/delay_analysis_tmp1.png')
plt.close()
