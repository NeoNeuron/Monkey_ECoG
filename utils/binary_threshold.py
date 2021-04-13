# Define types of get_threshold function for binary matrix reconstruction.

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from .utils import Linear_R2
from .roc import scan_auc_threshold
from .utils import Gaussian, Double_Gaussian

def get_linear_fit_pm(x_flatten:dict, y_flatten:dict, snr_mask:dict=None, is_log=True):
    pval = {}
    R2 = {}
    for band in y_flatten.keys():
        if is_log:
            log_y = np.log10(y_flatten[band])
        else:
            log_y = y_flatten[band].copy()
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

def find_gap_threshold(data_flatten):
    offset = int(len(data_flatten)/10)
    gap_th_label = None
    # Difference Gap
    data_sort = np.sort(data_flatten)
    data_diff = np.diff(data_sort)
    data_diff_sort = np.sort(data_diff[offset:-offset])
    if (data_diff_sort[-1] - data_diff_sort[-2]) > 2*data_diff_sort[1:].mean():
        max_id = np.argmax(data_diff[offset:-offset]) + offset
        th_val = (data_sort[max_id] + data_sort[max_id+1])/2
        gap_th_label = 'gap'
    else:
        kmean_toggle = False
        # Double Gaussian
        try:
            (counts, edges) = np.histogram(data_flatten, bins=100)
            popt, _ = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
            # find double Gaussian threshold
            if popt[2] > popt[3]:
                grid = np.arange(popt[3], popt[2], 0.001)
            else:
                grid = np.arange(popt[2], popt[3], 0.001)
            th_id = np.argmin(np.abs(Gaussian(grid, popt[0],popt[2],popt[4]) - Gaussian(grid, popt[1],popt[3],popt[5])))
            th_val = grid[th_id]
            if (popt[2] > edges[-1]) or popt[2] < edges[0] or popt[3] > edges[-1] or popt[3] < edges[0]:
                kmean_toggle = True
            else:
                gap_th_label = 'gauss'
        except:
            print(f'WARNING: Failed fitting.')
            kmean_toggle = True

        if kmean_toggle:
            # K-Means
            kmeans = KMeans(n_clusters=2).fit(data_flatten.reshape(-1, 1))
            if data_flatten[kmeans.labels_==0].mean() > data_flatten[kmeans.labels_==1].mean():
                label_large, label_small = 0, 1
            else:
                label_large, label_small = 1, 0
            kmean_th = (data_flatten[kmeans.labels_==label_large].min() + data_flatten[kmeans.labels_==label_small].max())/2
            th_val = kmean_th
            gap_th_label = 'kmean'
    return th_val, gap_th_label

# --------------------------------------------------
# Get threshold functions

def get_fit_threshold(weight_flatten:dict, prediction_flatten:dict, w_thresholds:np.ndarray, snr_mask:dict=None, is_log=True):
    pval,_ = get_linear_fit_pm(weight_flatten, prediction_flatten, snr_mask, is_log)
    fit_th = {}
    for band, p in pval.items():
        if is_log:
            fit_th[band] = np.array([10**(p[0]*i + p[1]) for i in np.log10(w_thresholds)])
        else:
            fit_th[band] = np.array([p[0]*i + p[1] for i in np.log10(w_thresholds)])
    return fit_th

def get_gap_threshold(data_flatten:dict, is_log=True):
    gap_th = {}
    for band,data_flatten_band in data_flatten.items():
        if is_log:
            gap_th[band],_ = find_gap_threshold(np.log10(data_flatten_band))
            gap_th[band] = 10**gap_th[band]
        else:
            gap_th[band],_ = find_gap_threshold(data_flatten_band)
    return gap_th

def get_roc_threshold(y_true:dict, y_predict:dict, w_thresholds:np.ndarray, is_log:bool=True):
    opt_th = {}
    for band, y_true_band in y_true.items():
        # TODO: add is_log flag
        _, opt_th[band] = scan_auc_threshold(y_predict[band], y_true_band, w_thresholds, is_log)
        if is_log:
            opt_th[band] = np.array([10**item for item in opt_th[band]])
    return opt_th