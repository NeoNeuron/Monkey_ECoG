#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from utils.tdmi import compute_snr_matrix
import pickle
from scipy.optimize import curve_fit
from utils.utils import Gaussian, Double_Gaussian
from sklearn.cluster import KMeans

path = 'tdmi_snr_analysis/'
# prepare weight_flatten
snr_th_manual = {
    'delta'      :3.5,
    'theta'      :5.0,
    'alpha'      :5.0,
    'beta'       :6.5,
    'gamma'      :20,  
    'high_gamma' :20,  
    'raw'        :8.0,
}
tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
off_diag_mask = ~np.eye(tdmi_data['raw'].shape[0], dtype=bool)
filter_pool = list(tdmi_data.files)

snr_th_gauss = {}
snr_th_kmean = {}
fig, ax = plt.subplots(2, 4, figsize=(18,8), sharex='row')
ax = ax.reshape(-1)
for i, band in enumerate(filter_pool):
    snr_matrix = compute_snr_matrix(tdmi_data[band])
    snr_log = np.log10(snr_matrix[off_diag_mask])

    (counts, edges) = np.histogram(snr_log, bins=100)
    ax[i].bar(edges[1:], counts, width=edges[1]-edges[0], alpha=.75)
    ax[i].set_title(band)
    ax[0].set_xlabel(r'$\log_{10}$(SNR)')
    ax[0].set_ylabel('Counts')

    try:
        popt, _ = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
        if popt[2] > popt[3]:
            ax[i].plot(edges[1:], Gaussian(edges[1:], popt[0],popt[2],popt[4]), 'ro', markersize = 3, label=r'$1^{st}$ Gaussian fit')
            ax[i].plot(edges[1:], Gaussian(edges[1:], popt[1],popt[3],popt[5]), 'bo', markersize = 3, label=r'$2^{nd}$ Gaussian fit')
        else:
            ax[i].plot(edges[1:], Gaussian(edges[1:], popt[0],popt[2],popt[4]), 'bo', markersize = 3, label=r'$2^{nd}$ Gaussian fit')
            ax[i].plot(edges[1:], Gaussian(edges[1:], popt[1],popt[3],popt[5]), 'ro', markersize = 3, label=r'$1^{nd}$ Gaussian fit')
        # find double Gaussian threshold
        if popt[2] > popt[3]:
            grid = np.arange(popt[3], popt[2], 0.001)
        else:
            grid = np.arange(popt[2], popt[3], 0.001)
        th_id = np.argmin(np.abs(Gaussian(grid, popt[0],popt[2],popt[4]) - Gaussian(grid, popt[1],popt[3],popt[5])))
        snr_th_gauss[band] = 10**grid[th_id]
        ax[i].axvline(grid[th_id], color = 'm', label='Double Gaussian th')

    except:
        print(f'WARNING: Failed fitting the {band:s} band case.')
        snr_th_gauss[band] = np.nan
        pass

    ax[i].axvline(np.log10(snr_th_manual[band]), color = 'orange', label='manual th')

    kmeans = KMeans(n_clusters=2).fit(snr_log.reshape(-1, 1))
    if snr_log[kmeans.labels_==0].mean() > snr_log[kmeans.labels_==1].mean():
        label_large, label_small = 0, 1
    else:
        label_large, label_small = 1, 0
    kmean_th = (snr_log[kmeans.labels_==label_large].min() + snr_log[kmeans.labels_==label_small].max())/2
    snr_th_kmean[band] = 10**kmean_th
    ax[i].axvline(kmean_th, color = 'cyan', label='kmean th')
# plot legend in the empty subplot
handles, labels = ax[0].get_legend_handles_labels()
ax[-1].legend(handles, labels, loc=2, fontsize=16)
ax[-1].axis('off')

plt.tight_layout()
fig.savefig(path + 'snr_dist_figure.png')

# save snr-th
with open(path + 'snr_th_manual.pkl', 'wb') as f:
    pickle.dump(snr_th_manual, f)
with open(path + 'snr_th_gauss.pkl', 'wb') as f:
    pickle.dump(snr_th_gauss, f)
with open(path + 'snr_th_kmean.pkl', 'wb') as f:
    pickle.dump(snr_th_kmean, f)
