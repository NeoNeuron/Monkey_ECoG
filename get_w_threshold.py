import numpy as np
from utils.utils import Linear_R2

def linear_fit(tdmi_data_flatten:dict, weight_flatten:dict):
    pval = {}
    R2 = {}
    for band in tdmi_data_flatten.keys():
        log_tdmi_data = np.log10(tdmi_data_flatten[band])
        answer = weight_flatten[band].copy()
        answer[answer==0]=1e-7
        log_answer = np.log10(answer)
        answer_edges = np.linspace(-6, 1, num = 100)
        # average data
        log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
        for i in range(len(answer_edges)-1):
            mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
            if mask.sum() == 0:
                log_tdmi_data_mean[i] = np.nan
            else:
                log_tdmi_data_mean[i] = log_tdmi_data[mask].mean()
        pval[band] = np.polyfit(
            answer_edges[:-1][~np.isnan(log_tdmi_data_mean)], 
            log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], 
            deg=1
        )
        R2[band] = Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval[band])
    return pval, R2

if __name__ == '__main__':
    from utils.tdmi import MI_stats
    path = 'tdmi_snr_analysis/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)

    # setup interarea mask
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    weight_flatten = {band:weight_flatten for band in filter_pool}
    tdmi_data_flatten = {}

    for band in filter_pool:
        tdmi_data_flatten[band] = MI_stats(tdmi_data[band], 'max')
        tdmi_data_flatten[band] = tdmi_data_flatten[band][~np.eye(stride[-1], dtype=bool)]

    pval, R2 = linear_fit(tdmi_data_flatten, weight_flatten)

    np.savez(path + 'pval.npz', **pval)
    np.savez(path + 'R2.npz', **R2)