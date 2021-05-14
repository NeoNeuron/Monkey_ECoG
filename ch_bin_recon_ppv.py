# Author: Kai Chen
# Description: Plot PPV figure for binary reconstruction results.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from utils.plot_frame import *
    plt.rcParams['lines.linewidth'] = 0.5
    from utils.plot import plot_ppv_curve
    path = 'tdmi_snr_analysis/'
    fnames = ['recon_fit_tdmi.npy', 'recon_gap_tdmi.npy', 'recon_roc_tdmi.npy']
    all_data = [np.load(path+fname, allow_pickle=True) for fname in fnames]
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw', 'sub_delta', 'above_delta']
    data_plt = {}
    for i, band in enumerate(filters):
        data_plt[band] = {
            'roc_data':[ data[:, i, :] for data in all_data ],
            'colors':['r','royalblue','orange'],
            'labels':[r'PPV(th$_{fit}$)', r'PPV(th$_{gap}$)', r'PPV(th$_{roc}$)'],
            'band':band,
        }
    
    fig = fig_frame52(data_plt, plot_ppv_curve)
    fig.savefig(path + 'ch_bin_recon_ppv.png')