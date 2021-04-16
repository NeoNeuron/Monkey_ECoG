# Author: Kai Chen
# Description: Binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;

if __name__ == '__main__':
    import pickle
    from utils.core import EcogTDMI
    from utils.binary_threshold import gen_bin_recon
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    data = EcogTDMI('data/')
    data.init_data(path)
    sc, fc = data.get_sc_fc('cg')
    roi_mask = data.roi_mask.copy() # ! excute aftesc_fc_sc_fc()
    # ==================================================
    weight = sc[data.filters[0]].copy()

    ifnames = ['th_fit_tdmi_CG.pkl', 'th_roc_tdmi_CG.pkl', 'th_gap_tdmi_CG.pkl']
    ofnames = ['recon_fit_tdmi_CG.pkl', 'recon_roc_tdmi_CG.pkl', 'recon_gap_tdmi_CG.pkl']
    for ifname, ofname in zip(ifnames, ofnames):
        with open(path+ifname, 'rb') as f:
            fc_th = pickle.load(f)
        sc_mask, fc_mask = gen_bin_recon(weight, fc, fc_th)
        with open(path + ofname, 'wb') as f:
            pickle.dump(sc_mask, f)
            pickle.dump(fc_mask, f)
            pickle.dump(roi_mask, f)