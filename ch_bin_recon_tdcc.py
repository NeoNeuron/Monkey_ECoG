# Author: Kai Chen
# Description: Binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;

if __name__ == '__main__':
    from fcpy.core import EcogTDCC
    from fcpy.binary_threshold import gen_bin_recon
    import pickle
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    data = EcogTDCC('data/')
    data.init_data(path, 'snr_th_gauss_tdcc.pkl') # for binary reconstruction, no need for SNR masking
    sc, fc = data.get_sc_fc('ch')
    roi_mask = data.roi_mask.copy() # ! excute after get_snr_mask()
    # ==================================================
    weight = sc[data.filters[0]].copy()

    ifnames = ['th_fit_tdcc.pkl', 'th_roc_tdcc.pkl', 'th_gap_tdcc.pkl']
    ofnames = ['recon_fit_tdcc.pkl', 'recon_roc_tdcc.pkl', 'recon_gap_tdcc.pkl']
    for ifname, ofname in zip(ifnames, ofnames):
        with open(path+ifname, 'rb') as f:
            fc_th = pickle.load(f)
        sc_mask, fc_mask = gen_bin_recon(weight, fc, fc_th)
        with open(path + ofname, 'wb') as f:
            pickle.dump(sc_mask, f)
            pickle.dump(fc_mask, f)
            pickle.dump(roi_mask, f)