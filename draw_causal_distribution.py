#!/usr/bin python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=16
import matplotlib.pyplot as plt

data_package = np.load('preprocessed_data.npz')

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

for band in filter_pool:
    tdmi_data = np.load('sum_tdmi_processing_1217/data/data_r_'+band+'_tdmi_126-10_total.npy')
    print(tdmi_data.shape)
    # sum tdmi mode
    # tdmi_data = tdmi_data[:,:,:10].sum(2)
    # max tdmi mode
    tdmi_data = tdmi_data.max(2)
    log_tdmi_data = np.log10(tdmi_data)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # calculate histogram
    (counts, edges) = np.histogram(np.log10(tdmi_data.flatten()), bins=100, density=True)

    fig, ax = plt.subplots(2,4,figsize=(20,10))

    def Gaussian(x, a, mu, sigma):
        return a*np.exp(-(x-mu)**2/sigma)

    def Double_Gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2):
        return Gaussian(x, a1, mu1, sigma1) + Gaussian(x, a2, mu2, sigma2)

    from scipy.optimize import curve_fit


    ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
    try:
        popt, pcov = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
        ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[0],popt[2],popt[4]), 'ro', markersize = 4, label=r'$1^{st}$ Gaussian fit')
        ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[1],popt[3],popt[5]), 'bo', markersize = 4, label=r'$2^{nd}$ Gaussian fit')
    except:
        print(f'WARNING: Failed fitting the {band:s} band case.')
        pass
    ax[0,0].set_xlabel('$log_{10}(Value)$')
    ax[0,0].set_ylabel('Density')
    ax[0,0].legend(fontsize=15)

    answer = data_package['con_known']
    # pval, cov = np.polyfit(answer.flatten(), log_tdmi_data.flatten(), deg=1,cov=True)
    answer_set = np.unique(answer.flatten())
    log_tdmi_data_mean = np.array([np.mean(log_tdmi_data.flatten()[answer.flatten()==key]) for key in answer_set])
    answer_set[answer_set==0]=1e-6
    pval, cov = np.polyfit(np.log10(answer_set), log_tdmi_data_mean, deg=1,cov=True)
    ax[1,0].plot(np.log10(answer.flatten()), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[1,0].plot(np.log10(answer_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
    ax[1,0].plot(np.log10(answer_set), np.polyval(pval, np.log10(answer_set)), 'r', label='Linear Fitting')
    ax[1,0].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
    ax[1,0].set_xlabel(r'$log_{10}$(Connectivity Strength)')
    ax[1,0].set_title(f'Fitting Slop = {pval[0]:5.3f}')
    ax[1,0].legend(fontsize=15)

    threshold_options = [0.1, 1e-3, 1e-5]
    for idx, threshold in enumerate(threshold_options):
        answer = data_package['con_known']
        ax[0,idx+1].semilogy(np.sort(answer.flatten()))
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
    plt.savefig('data_r_'+band+'_analysis.png')
