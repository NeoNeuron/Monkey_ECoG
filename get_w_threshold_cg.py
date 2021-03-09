
if __name__ == '__main__':
    from utils.tdmi import MI_stats, compute_snr_matrix
    from utils.utils import CG
    from get_w_threshold import linear_fit
    import numpy as np
    import pickle
    path = 'tdmi_snr_analysis/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)

    # load and manipulate weight matrix
    weight = data_package['adj_mat']
    weight[weight == 0] = 1e-6
    cg_mask = np.diag(multiplicity == 1).astype(bool)
    weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
    weight[cg_mask] = np.nan

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data = np.load(path+'tdmi_data_long.npz', allow_pickle=True)

    # setup interarea mask
    weight_flatten = weight[~cg_mask]
    weight_flatten = {band:weight_flatten for band in filter_pool}
    tdmi_data_flatten = {}
    snr_mask = {}
    with open(path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)

    for band in filter_pool:
        snr_mat = compute_snr_matrix(tdmi_data[band])
        snr_mask = snr_mat >= snr_th[band]
        tdmi_data_band = MI_stats(tdmi_data[band], 'max')
        tdmi_data_band[~snr_mask] = np.nan
        tdmi_data_cg = CG(tdmi_data_band, stride)
        tdmi_data_flatten[band] = tdmi_data_cg[~cg_mask]

    pval, R2 = linear_fit(tdmi_data_flatten, weight_flatten)

    np.savez(path + 'pval_cg.npz', **pval)
    np.savez(path + 'R2_cg.npz', **R2)