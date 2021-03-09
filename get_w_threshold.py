from utils.tdmi import compute_snr_matrix
import numpy as np
from utils.utils import Linear_R2

def linear_fit(tdmi_data_flatten:dict, weight_flatten:dict, snr_mask:dict=None):
    pval = {}
    R2 = {}
    for band in tdmi_data_flatten.keys():
        log_tdmi_data = np.log10(tdmi_data_flatten[band])
        if snr_mask is not None:
            log_tdmi_data[~snr_mask[band]] = np.nan
        answer = weight_flatten[band].copy()
        answer[answer==0]=1e-7
        log_answer = np.log10(answer)
        answer_edges = np.linspace(-6, 1, num = 15)
        answer_center = (answer_edges[1:] + answer_edges[:-1])/2
        # average data
        log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
        for i in range(len(answer_edges)-1):
            mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
            if mask.sum() == 0:
                log_tdmi_data_mean[i] = np.nan
            else:
                log_tdmi_data_mean[i] = np.nanmean(log_tdmi_data[mask])
        pval[band] = np.polyfit(
            answer_center[~np.isnan(log_tdmi_data_mean)], 
            log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], 
            deg=1
        )
        R2[band] = Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval[band])
    return pval, R2

if __name__ == '__main__':
    from utils.tdmi import MI_stats
    from utils.tdmi import compute_snr_matrix
    import pickle
    path = 'tdmi_snr_analysis/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    weight = data_package['weight']
    off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data = np.load(path+'tdmi_data_long.npz', allow_pickle=True)

    # setup interarea mask
    weight_flatten = weight[off_diag_mask]
    weight_flatten = {band:weight_flatten for band in filter_pool}
    tdmi_data_flatten = {}
    snr_mask = {}
    with open(path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)

    for band in filter_pool:
        tdmi_data_flatten[band] = MI_stats(tdmi_data[band], 'max')
        tdmi_data_flatten[band] = tdmi_data_flatten[band][off_diag_mask]
        snr_mat = compute_snr_matrix(tdmi_data[band])
        snr_mask[band] = snr_mat >= snr_th[band]
        snr_mask[band] = snr_mask[band][off_diag_mask]

    pval, R2 = linear_fit(tdmi_data_flatten, weight_flatten, snr_mask)

    np.savez(path + 'pval.npz', **pval)
    np.savez(path + 'R2.npz', **R2)