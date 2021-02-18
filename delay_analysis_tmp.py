import os
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from draw_causal_distribution_v2 import load_data
from tdmi_delay_analysis_v1 import get_delay_matrix


path = 'data_preprocessing_46_region/'
band = 'raw'
tdmi_data = load_data(path, band)
n_channel = tdmi_data.shape[0]
n_delay = tdmi_data.shape[2]
# complete the tdmi series
tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
tdmi_data_full[:,:,n_delay-1:] = tdmi_data
tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)
delay_mat = get_delay_matrix(path, band, force_compute=True)
fig, ax = plt.subplots(4,1, figsize=(10,12), sharex=True)
[ax[0].plot(np.arange(2*n_delay-1)-n_delay, curve, alpha=.6, color='r') for curve in tdmi_data_full[delay_mat>200]]
[ax[1].plot(np.arange(2*n_delay-1)-n_delay, curve, alpha=.6, color='b') for curve in tdmi_data_full[delay_mat<-200]]
[ax[2].plot(np.arange(2*n_delay-1)-n_delay, curve, alpha=.6, color='g') for curve in tdmi_data_full[delay_mat==0]]
[ax[3].plot(np.arange(2*n_delay-1)-n_delay, curve, alpha=.6, color='grey') for curve in tdmi_data_full[(delay_mat!=0)*(delay_mat>=-200)*(delay_mat<=200)]]
plt.tight_layout()
plt.savefig('tmp/delay_analysis_tmp.png')
plt.close()
