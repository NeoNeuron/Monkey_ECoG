# Author: Kai Chen
# Description: Binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;
import numpy as np

def gen_bin_recon(weight, fc, fc_th):
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    sc_mask = [weight > w_th for w_th in w_thresholds]
    fc_mask = {}
    for band in fc.keys():
        if isinstance(fc_th[band], np.ndarray):
            fc_mask[band] = np.array([(fc[band] >= fc_th_i) for fc_th_i in fc_th[band]])
        else:
            fc_mask[band] = (fc[band] >= fc_th[band])
    return sc_mask, fc_mask

if __name__ == '__main__':
    from utils.core import EcogGC
    import pickle
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    data = EcogGC('data/')
    data.init_data() # for binary reconstruction, no need for SNR masking
    sc, fc = data.get_sc_fc('ch')
    roi_mask = data.roi_mask.copy() # ! excute after get_snr_mask()
    # ==================================================
    weight = sc[data.filters[0]].copy()

    ifnames = ['th_fit_gc.pkl', 'th_roc_gc.pkl', 'th_gap_gc.pkl']
    ofnames = ['recon_fit_gc.pkl', 'recon_roc_gc.pkl', 'recon_gap_gc.pkl']
    for ifname, ofname in zip(ifnames, ofnames):
        with open(path+ifname, 'rb') as f:
            fc_th = pickle.load(f)
        sc_mask, fc_mask = gen_bin_recon(weight, fc, fc_th)
        with open(path + ofname, 'wb') as f:
            pickle.dump(sc_mask, f)
            pickle.dump(fc_mask, f)
            pickle.dump(roi_mask, f)