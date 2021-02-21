import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from utils.tdmi import compute_tdmi_full, compute_delay_matrix


path = 'data_preprocessing_46_region/'
band = 'raw'
tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
tdmi_data_full = compute_tdmi_full(tdmi_data[band])
delay_mat = compute_delay_matrix(tdmi_data[band])
n_delay = tdmi_data[band].shape[2]
fig, ax = plt.subplots(4,1, figsize=(10,12), sharex=True)
[ax[0].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='r') for curve in tdmi_data_full[delay_mat>200]]
[ax[1].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='b') for curve in tdmi_data_full[delay_mat<-200]]
[ax[2].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='g') for curve in tdmi_data_full[delay_mat==0]]
[ax[3].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='grey') for curve in tdmi_data_full[(delay_mat!=0)*(delay_mat>=-200)*(delay_mat<=200)]]
plt.tight_layout()
plt.savefig('tmp/delay_analysis_tmp.png')
plt.close()
