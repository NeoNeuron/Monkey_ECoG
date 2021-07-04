# Author: Kai Chen
# Description: Convert binary reconstruction results to markdown format.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;

if __name__ == '__main__':
    from fcpy.utils import pkl2md
    import pickle
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    fnames = ['recon_fit_gc.pkl', 'recon_roc_gc.pkl', 'recon_gap_gc.pkl']
    # ==================================================

    for fname in fnames:
        with open(path+fname, 'rb') as f:
            sc_mask = pickle.load(f)
            fc_mask = pickle.load(f)
            pkl2md(path + fname.replace('pkl', 'md'), sc_mask, fc_mask)