# Author: Kai Chen
# Description: Plot binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;
from utils.plot import gen_binary_recon_figure
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.1

def gen_figures(fname:str, sc_mask:list, fc_mask:dict, roi_mask):
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    for key, item in fc_mask.items():
        if item is not None:
            if len(item.shape) == 1:
                fc_mask[key] = np.tile(item, (len(w_thresholds),1))
    fc_mask_buffer = [{}]*len(w_thresholds)
    for band, item in fc_mask.items():
        if item is not None:
            for idx in range(len(w_thresholds)):
                fc_mask_buffer[idx][band] = item[idx]
        else:
            for idx in range(len(w_thresholds)):
                fc_mask_buffer[idx][band] = None

    for idx in range(len(w_thresholds)):
        fig = gen_binary_recon_figure(fc_mask_buffer[idx], sc_mask[idx], roi_mask)
        plt.tight_layout()
        fig.savefig(fname.replace('.pkl', f'_{idx:d}.png'))
        plt.close()

if __name__ == '__main__':
    import pickle
    path = 'tdmi_snr_analysis/'
    # ==================================================
    fnames = ['recon_fit_tdmi.pkl', 'recon_roc_tdmi.pkl', 'recon_gap_tdmi.pkl']
    # ==================================================

    for fname in fnames:
        with open(path+fname, 'rb') as f:
            sc_mask = pickle.load(f)
            fc_mask = pickle.load(f)
            roi_mask = pickle.load(f)
            gen_figures(path + fname, sc_mask, fc_mask, roi_mask)