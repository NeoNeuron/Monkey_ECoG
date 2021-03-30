# Author: Kai Chen
# Description: Binary reconstruction of adjacent matrix.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;
from utils.core import EcogTDMI
import numpy as np

def gen_bin_recon(weight, fc_th):
    sc_mask = [weight > w_th for w_th in w_thresholds]
    fc_mask = {}
    for band in data.filters:
        if isinstance(fc_th[band], np.ndarray):
            fc_mask[band] = np.array([(fc[band] > fc_th_i) for fc_th_i in fc_th[band]])
        else:
            fc_mask[band] = (fc[band] > fc_th[band])
    return sc_mask, fc_mask

if __name__ == '__main__':
    import pickle
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    data = EcogTDMI('data/')
    data.init_data(path)
    sc, fc = data.get_sc_fc('cg')
    roi_mask = data.roi_mask.copy() # ! excute aftesc_fc_sc_fc()
    # ==================================================
    weight = sc[data.filters[0]].copy()

    w_thresholds = np.logspace(-6, 0, num=7, base=10)

    ifnames = ['th_fit_tdmi_CG.pkl', 'th_roc_tdmi_CG.pkl', 'th_gap_tdmi_CG.pkl']
    ofnames = ['recon_fit_tdmi_CG.pkl', 'recon_roc_tdmi_CG.pkl', 'recon_gap_tdmi_CG.pkl']
    for ifname, ofname in zip(ifnames, ofnames):
        with open(path+ifname, 'rb') as f:
            fc_th = pickle.load(f)
        sc_mask, fc_mask = gen_bin_recon(weight, fc_th)
        with open(path + ofname, 'wb') as f:
            pickle.dump(sc_mask, f)
            pickle.dump(fc_mask, f)
            pickle.dump(roi_mask, f)