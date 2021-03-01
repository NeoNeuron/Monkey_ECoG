import numpy as np
import matplotlib.pyplot as plt
from utils.tdmi import compute_snr_matrix, compute_delay_matrix

path = 'tdmi_snr_analysis/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
weight = data_package['weight']
tdmi_data = np.load(path + 'tdmi_data.npz', allow_pickle=True)
# manually set snr threshold
snr_th = {
    'raw'        :5.0,
    'delta'      :4.3,
    'theta'      :4.5,
    'alpha'      :4.,
    'beta'       :5.,
    'gamma'      :11,
    'high_gamma' :11,
}

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
indices = [(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,1),]
fig, ax = plt.subplots(4,2, figsize=(7,12))
for band, index in zip(filter_pool, indices):
    delay_matrix = compute_delay_matrix(tdmi_data[band])
    delay_mask = np.abs(delay_matrix)<= 200
    snr_matrix = compute_snr_matrix(tdmi_data[band])
    snr_mask = snr_matrix >= snr_th[band]

    ax[index].semilogx(weight[snr_mask*delay_mask].flatten(), delay_matrix[snr_mask*delay_mask].flatten(), '.k', ms=0.5)
    ax[index].set_xlabel('Weight')
    ax[index].set_ylabel('Delay (ms)')
    ax[index].set_title(band)
ax[0,0].set_visible(False)
plt.tight_layout()
plt.savefig(path + 'weight_delay.png')
