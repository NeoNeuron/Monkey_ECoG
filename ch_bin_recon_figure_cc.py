# Author: Kai Chen
# Description: Plot binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.1

if __name__ == '__main__':
    import pickle
    from ch_bin_recon_figure import gen_figures
    path = 'tdmi_snr_analysis/'
    # ==================================================
    fnames = ['recon_fit_cc.pkl', 'recon_roc_cc.pkl', 'recon_gap_cc.pkl']
    # ==================================================

    for fname in fnames:
        with open(path+fname, 'rb') as f:
            sc_mask = pickle.load(f)
            fc_mask = pickle.load(f)
            roi_mask = pickle.load(f)
            gen_figures(path + fname, sc_mask, fc_mask, roi_mask)