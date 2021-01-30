#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

from math import log
import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define fitting function
def Gaussian(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/sigma)

def Double_Gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2):
    return Gaussian(x, a1, mu1, sigma1) + Gaussian(x, a2, mu2, sigma2)

path = 'data_preprocessing_46_region/'
data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

tdmi_mode = 'sum'     # or 'max'
is_interarea = False  # is inter area or not

for band in filter_pool:
    # setup interarea mask
    weight_flatten = data_package['weight_flatten'].copy()
    if is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]

    if band == None:
        tdmi_data = np.load(path + 'data_series_tdmi_total.npy', allow_pickle=True)
        tdmi_data_shuffle = np.load(path + 'data_series_tdmi_shuffle.npy', allow_pickle=True)
    else:
        tdmi_data = np.load(path + 'data_series_'+band+'_tdmi_total.npy', allow_pickle=True)
        tdmi_data_shuffle = np.load(path + 'data_series_'+band+'_tdmi_shuffle.npy', allow_pickle=True)
    tdmi_data_flatten = []
    for i in range(tdmi_data.shape[0]):
        for j in range(tdmi_data.shape[1]):
            if tdmi_mode == 'sum':
                tdmi_data[i,j] = tdmi_data[i,j][:,:,1:11].sum(2)
            elif tdmi_mode == 'max':
                tdmi_data[i,j] = tdmi_data[i,j].max(2)
            else:
                raise RuntimeError('Invalid tdmi_mode.')
            if i != j:
                tdmi_data_flatten.append(tdmi_data[i,j].flatten())
            else:
                tdmi_data_flatten.append(tdmi_data[i,j][~np.eye(data_package['multiplicity'][i], dtype=bool)])

    tdmi_data_flatten = np.hstack(tdmi_data_flatten)
    if is_interarea:
        log_tdmi_data = np.log10(tdmi_data_flatten[interarea_mask])
    else:
        log_tdmi_data = np.log10(tdmi_data_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # calculate histogram
    (counts, edges) = np.histogram(log_tdmi_data, bins=100, density=True)

    fig, ax = plt.subplots(2,4,figsize=(20,10))

    SI_value = tdmi_data_shuffle[~np.eye(data_package['multiplicity'].sum(), dtype=bool)].mean()
    if tdmi_mode == 'sum':
        SI_value *= 10
    ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
    ax[0,0].axvline(np.log10(SI_value), color='cyan')
    # try:
    #     popt, pcov = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
    #     ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[0],popt[2],popt[4]), 'ro', markersize = 4, label=r'$1^{st}$ Gaussian fit')
    #     ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[1],popt[3],popt[5]), 'bo', markersize = 4, label=r'$2^{nd}$ Gaussian fit')
    # except:
    #     print(f'WARNING: Failed fitting the {band:s} band case.')
    #     pass
    ax[0,0].set_xlabel('$log_{10}(Value)$')
    ax[0,0].set_ylabel('Density')
    ax[0,0].legend(fontsize=15)

    # pval, cov = np.polyfit(answer.flatten(), log_tdmi_data.flatten(), deg=1,cov=True)
    weight_set = np.unique(weight_flatten)
    log_tdmi_data_mean = np.array([np.mean(log_tdmi_data[weight_flatten==key]) for key in weight_set])
    weight_set[weight_set==0]=1e-6
    pval, cov = np.polyfit(np.log10(weight_set), log_tdmi_data_mean, deg=1,cov=True)
    ax[1,0].plot(np.log10(weight_flatten+1e-8), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[1,0].plot(np.log10(weight_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
    ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
    if tdmi_mode == 'sum':
        ax[1,0].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
    elif tdmi_mode == 'max':
        ax[1,0].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
    ax[1,0].set_xlabel(r'$log_{10}$(Connectivity Strength)')
    ax[1,0].set_title(f'Fitting Slop = {pval[0]:5.3f}')
    ax[1,0].legend(fontsize=15)

    # Draw ROC curves
    threshold_options = [1e-1, 5e-3, 1e-4]
    for idx, threshold in enumerate(threshold_options):
        answer = weight_flatten.copy()
        ax[0,idx+1].semilogy(np.sort(answer))
        ax[0,idx+1].set_xlabel('Ranked connectivity strength')
        ax[0,idx+1].set_ylabel('Connectivity Strength')
        ax[0,idx+1].axhline(threshold, color='r', ls = '--')
        ax[0,idx+1].set_title(f'Threshold = {threshold:3.2e}')

        # Plot ROC curve
        answer = (answer>threshold).astype(bool)

        false_positive = np.array([np.sum((log_tdmi_data>i)*(~answer))/np.sum(~answer) for i in np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)])
        true_positive = np.array([np.sum((log_tdmi_data>i)*(answer))/np.sum(answer) for i in np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)])

        ax[1,idx+1].plot(false_positive, true_positive)
        ax[1,idx+1].set_xlabel('False positive rate')
        ax[1,idx+1].set_ylabel('True positive rate')
        ax[1,idx+1].plot(range(2),range(2), '--')
        ax[1,idx+1].set_xlim(0,1)
        ax[1,idx+1].set_ylim(0,1)
        auc = -np.sum(np.diff(false_positive)*(true_positive[:-1]+true_positive[1:])/2)
        ax[1,idx+1].set_title(f'AUC = {auc:5.3f}')

    plt.tight_layout()
    if band == None:
        if is_interarea:
            plt.savefig(path + 'channel_interarea_analysis_'+tdmi_mode+'.png')
        else:
            plt.savefig(path + 'channel_analysis_'+tdmi_mode+'.png')
    else:
        if is_interarea:
            plt.savefig(path + 'channel_'+band+'_interarea_analysis_'+tdmi_mode+'.png')
        else:
            plt.savefig(path + 'channel_'+band+'_analysis_'+tdmi_mode+'.png')