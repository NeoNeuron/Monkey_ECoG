# Define types of get_threshold function for binary matrix reconstruction.

import numpy as np
from .utils import Linear_R2
from .roc import scan_auc_threshold

def get_linear_fit_pm(x_flatten:dict, y_flatten:dict, snr_mask:dict=None):
    pval = {}
    R2 = {}
    for band in y_flatten.keys():
        log_y = np.log10(y_flatten[band])
        if snr_mask is not None:
            log_y[~snr_mask[band]] = np.nan
        answer = x_flatten[band].copy()
        answer[answer==0]=1e-7 # to avoid log10(0)=nan
        log_answer = np.log10(answer)
        answer_edges = np.linspace(-6, 1, num = 15)
        answer_center = (answer_edges[1:] + answer_edges[:-1])/2
        # average data
        log_y_mean = np.zeros(len(answer_edges)-1)
        for i in range(len(answer_edges)-1):
            mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
            if mask.sum() == 0:
                log_y_mean[i] = np.nan
            else:
                log_y_mean[i] = np.nanmean(log_y[mask])
        pval[band] = np.polyfit(
            answer_center[~np.isnan(log_y_mean)], 
            log_y_mean[~np.isnan(log_y_mean)], 
            deg=1
        )
        R2[band] = Linear_R2(answer_edges[:-1], log_y_mean, pval[band])
    return pval, R2

def find_gap_threshold(data_flatten, offset=1000):
    tdmi_sort = np.sort(data_flatten)
    max_id = np.argmax(np.diff(tdmi_sort)[offset:-offset]) + offset
    th_val = (tdmi_sort[max_id] + tdmi_sort[max_id+1])/2
    return th_val

# --------------------------------------------------
# Get threshold functions

def get_fit_threshold(weight_flatten:dict, prediction_flatten:dict, w_thresholds:np.ndarray, snr_mask:dict=None):
    pval,_ = get_linear_fit_pm(weight_flatten, prediction_flatten, snr_mask)
    fit_th = {}
    for band, p in pval.items():
        fit_th[band] = np.array([10**(p[0]*i + p[1]) for i in np.log10(w_thresholds)])
    return fit_th

def get_gap_threshold(data_flatten:dict, offset:int=1000):
    gap_th = {}
    for band,data_flatten_band in data_flatten.items():
        gap_th[band] = find_gap_threshold(np.log10(data_flatten_band), offset)
        gap_th[band] = 10**gap_th[band]
    return gap_th

def get_roc_threshold(y_true:dict, y_predict:dict, w_thresholds:np.ndarray):
    opt_th = {}
    for band, y_true_band in y_true.items():
        _, opt_th[band] = scan_auc_threshold(y_predict[band], y_true_band, w_thresholds)
        opt_th[band] = np.array([10**item for item in opt_th[band]])
    return opt_th