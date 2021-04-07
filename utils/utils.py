import numpy as np
import time
from .roc import ROC_matrix

def Gaussian(x, a, mu, sigma):
    """Standand Gaussian template function.

    Args:
        x (array-like): 1-d input data.
        a (float): positive contrained Gaussian amplitude
        mu (float): mean
        sigma (float): variance

    Returns:
        array-like: function value
    """
    return np.abs(a)*np.exp(-(x-mu)**2/sigma)

def Double_Gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2):
    """Double Gaussian like template function.

    Args:
        x (array-like): 1d input data
        a1 (float): amplitude of first Gaussian
        a2 (float): amplitude of second Gaussian
        mu1 (float): mean of first Gaussian
        mu2 (float): mean of second Gaussian
        sigma1 (float): variance of first Gaussian
        sigma2 (float): variance of second Gaussian

    Returns:
        array-like: function value
    """
    return Gaussian(x, a1, mu1, sigma1) + Gaussian(x, a2, mu2, sigma2)

def print_log(string, t0):
    """Print log info.

    Args:
        string (str): string-like information to print.
        t0 (float): time stamp for starting of program.
    """
    print(f"[INFO] {time.time()-t0:6.2f}: {string:s}")

def Linear_R2(x:np.ndarray, y:np.ndarray, pval:np.ndarray)->float:
    """Compute R-square value for linear fitting.

    Args:
        x (np.ndarray): variable of function
        y (np.ndarray): image of function
        pval (np.ndarray): parameter of linear fitting

    Returns:
        float: R square value
    """
    mask = ~np.isnan(x)*~np.isnan(y)*~np.isinf(x)*~np.isinf(y)# filter out nan
    y_predict = x[mask]*pval[0]+pval[1]
    R = np.corrcoef(y[mask], y_predict)[0,1]
    return R**2

def linear_fit(x, y):
    not_nan_mask = ~np.isnan(x)*~np.isnan(y)*~np.isinf(x)
    pval = np.polyfit(x[not_nan_mask], y[not_nan_mask], deg=1)
    r = Linear_R2(x, y, pval)**0.5
    return pval, r

def CG(tdmi_data:np.ndarray, stride:np.ndarray)->np.ndarray:
    """Compute the coarse-grained average of 
        each cortical region for tdmi_data.

    Args:
        tdmi_data (np.ndarray): channel-wise tdmi_data.
        stride (np.ndarray): stride of channels. 
            Equal to the `cumsum` of multiplicity.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data
    """
    multiplicity = np.diff(stride).astype(int)
    n_region = stride.shape[0]-1
    tdmi_data_cg = np.zeros((n_region, n_region))
    for i in range(n_region):
        for j in range(n_region):
            data_buffer = tdmi_data[stride[i]:stride[i+1],stride[j]:stride[j+1]]
            if i != j:
                tdmi_data_cg[i,j]=np.nanmean(data_buffer)
            else:
                if multiplicity[i] > 1:
                    tdmi_data_cg[i,j]=np.nanmean(data_buffer[~np.eye(multiplicity[i], dtype=bool)])
                else:
                    tdmi_data_cg[i,j]=np.nan # won't be used in ROC.
    return tdmi_data_cg

def pkl2md(fname:str, sc_mask:list, fc_mask:dict):
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    if len(list(fc_mask.values())[0].shape) == 1:
        for key, item in fc_mask.items():
            fc_mask[key] = np.tile(item, (len(w_thresholds),1))
    with open(fname, 'w') as ofile:
        roc_data = np.zeros((w_thresholds.shape[0], len(fc_mask.keys()), 8,))
        for idx, sc in enumerate(sc_mask):
            print("## $w_{ij}>10^{%d}$ " % int(np.log10(w_thresholds[idx])), file=ofile)
            print(f'p = {np.sum(sc)/sc.shape[0]:6.3f}', file=ofile)
            print('| band | TP | FP | FN | TN | Corr | TPR | FPR | PPV |', file=ofile)
            print('|------|----|----|----|----|------| --- | --- | --- |', file=ofile)

            union_mask = np.zeros_like(sc, dtype=bool)
            for iidx, band in enumerate(fc_mask.keys()):
                if band != 'raw':
                    union_mask += fc_mask[band][idx]
                TP, FP, FN, TN = ROC_matrix(sc, fc_mask[band][idx])
                CORR = np.corrcoef(sc, fc_mask[band][idx])[0, 1]
                if np.isnan(CORR):
                    CORR = 0.
                roc_data[idx, iidx, :] = [TP,FP,FN,TN,CORR,TP/(TP+FN),FP/(FP+TN),TP/(TP+FP)]
                print('|%s|%d|%d|%d|%d|%6.3f|%6.3f|%6.3f|%6.3f|' % (band, *roc_data[idx, iidx, :]), file=ofile)
            print(f'**CORR = {np.corrcoef(sc, union_mask)[0, 1]:6.3f}**', file=ofile)

    np.save(fname.replace('md', 'npy'), roc_data)