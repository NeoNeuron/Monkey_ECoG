# Author: Kai Chen
# Description: Plot binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;

if __name__ == '__main__':
    import pickle
    from utils.plot import gen_binary_recon_figures
    path = 'tdmi_snr_analysis/'
    # ==================================================
    fnames = ['recon_fit_tdmi.pkl', 'recon_roc_tdmi.pkl', 'recon_gap_tdmi.pkl']
    # ==================================================

    for fname in fnames:
        with open(path+fname, 'rb') as f:
            sc_mask = pickle.load(f)
            fc_mask = pickle.load(f)
            roi_mask = pickle.load(f)
            gen_binary_recon_figures(path + fname, sc_mask, fc_mask, roi_mask)