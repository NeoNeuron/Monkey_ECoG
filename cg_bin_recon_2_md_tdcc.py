# Author: Kai Chen
# Description: Convert binary reconstruction results to markdown format.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;
if __name__ == '__main__':
    import pickle
    from utils.utils import pkl2md
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    fnames = ['recon_fit_tdcc_CG.pkl', 'recon_roc_tdcc_CG.pkl', 'recon_gap_tdcc_CG.pkl']
    # ==================================================

    for fname in fnames:
        with open(path+fname, 'rb') as f:
            sc_mask = pickle.load(f)
            fc_mask = pickle.load(f)
            pkl2md(path + fname.replace('pkl', 'md'), sc_mask, fc_mask)