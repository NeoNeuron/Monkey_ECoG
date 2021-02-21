import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from utils.tdmi import compute_delay_matrix


path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
band = 'raw'
tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
n_delay = tdmi_data[band].shape[2]

delay_mat = compute_delay_matrix(tdmi_data[band])
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
