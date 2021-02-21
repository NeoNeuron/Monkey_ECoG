import numpy as np
import matplotlib.pyplot as plt
from .roc import ROC_curve, AUC, Youden_Index
from .utils import Linear_R2

def gen_causal_distribution_figure(tdmi_flatten:np.ndarray, 
                                   weight_flatten:np.ndarray,
                                   tdmi_threshold:float, 
                                   )->plt.Figure:
    """Generated figure for analysis of causal distributions.

    Args:
        tdmi_flatten (np.ndarray): flattened data for target tdmi statistics.
        weight_flatten (np.ndarray): flattened data for true connectome.
        tdmi_threshold (float): significance value of tdmi statistics.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    log_tdmi_data = np.log10(tdmi_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # calculate histogram
    (counts, edges) = np.histogram(log_tdmi_data, bins=100, density=True)

    # create figure canvas
    fig, ax = plt.subplots(2,4,figsize=(20,10))

    ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
    ax[0,0].axvline(np.log10(tdmi_threshold), color='cyan', label='SI')
    # UNCOMMENT to create double Gaussian fitting of TDMI PDF
    # from scipy.optimize import curve_fit
    # from .utils import Gaussian, Double_Gaussian
    # try:
    #     popt, _ = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
    #     ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[0],popt[2],popt[4]), 'ro', markersize = 4, label=r'$1^{st}$ Gaussian fit')
    #     ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[1],popt[3],popt[5]), 'bo', markersize = 4, label=r'$2^{nd}$ Gaussian fit')
    # except:
    #     print(f'WARNING: Failed fitting the {band:s} band case.')
    #     pass
    ax[0,0].set_xlabel('$log_{10}(Value)$')
    ax[0,0].set_ylabel('Density')
    ax[0,0].legend(fontsize=13, loc=2)

    weight_set = np.unique(weight_flatten)
    log_tdmi_data_mean = np.array([np.mean(log_tdmi_data[weight_flatten==key]) for key in weight_set])
    weight_set[weight_set==0]=1e-6
    pval = np.polyfit(np.log10(weight_set), log_tdmi_data_mean, deg=1)
    ax[1,0].plot(np.log10(weight_flatten+1e-8), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[1,0].plot(np.log10(weight_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
    ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
    ax[1,0].set_xlabel(r'$log_{10}$(Connectivity Strength)')
    ax[1,0].set_title(f'Fitting Slop = {pval[0]:5.3f}')
    ax[1,0].legend(fontsize=15)

    # Draw ROC curves
    threshold_options = [1e-1, 5e-3, 1e-4]
    opt_threshold = np.zeros(len(threshold_options))
    for idx, threshold in enumerate(threshold_options):
        answer = weight_flatten.copy()
        ax[0,idx+1].semilogy(np.sort(answer))
        ax[0,idx+1].set_xlabel('Ranked connectivity strength')
        ax[0,idx+1].set_ylabel('Connectivity Strength')
        ax[0,idx+1].axhline(threshold, color='r', ls = '--')
        ax[0,idx+1].set_title(f'Threshold = {threshold:3.2e}')

        # Plot ROC curve
        answer = (answer>threshold).astype(bool)
        thresholds = np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)
        fpr, tpr = ROC_curve(answer, log_tdmi_data, thresholds)
        Youden_index = Youden_Index(fpr, tpr)
        opt_threshold[idx] = thresholds[Youden_index]

        ax[1,idx+1].plot(fpr, tpr, 'navy')
        ax[1,idx+1].set_xlabel('False positive rate')
        ax[1,idx+1].set_ylabel('True positive rate')
        ax[1,idx+1].plot(range(2),range(2), '--', color='orange')
        ax[1,idx+1].set_xlim(0,1)
        ax[1,idx+1].set_ylim(0,1)
        ax[1,idx+1].set_title(f'AUC = {AUC(fpr, tpr):5.3f}')

    ax[0,0].axvline(opt_threshold.mean(), color='orange', label='opt_threshold')
    ax[0,0].legend(fontsize=13, loc=2)

    plt.tight_layout()
    return fig

def gen_auc_threshold_figure(aucs:dict, w_thresholds:list)->plt.Figure:
    fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
    ax = ax.reshape((8,))
    idx = 0
    for band, auc in aucs.items():
        if auc is None:
            ax[idx].set_visible(False)
        else:
            # plot dependence of AUC w.r.t w_threshold value
            ax[idx].semilogx(w_thresholds, auc, '-*', color='navy')
            ax[idx].grid(ls='--')
            ax[idx].set_title(band)
        idx += 1

    [ax[i].set_ylabel('AUC') for i in (0,4)]
    [ax[i].set_xlabel('Threshold value') for i in (4,5,6)]

    # make last subfigure invisible
    ax[-1].set_visible(False)
    plt.tight_layout()
    return fig

def gen_mi_s_figure(tdmi_data_flatten:dict, weight_flatten:dict)->plt.Figure:
    fig, ax = plt.subplots(2,4,figsize=(20,10))
    ax = ax.reshape((8,))
    idx = 0
    for band, tdmi_flatten in tdmi_data_flatten.items():
        if tdmi_flatten is None:
            ax[idx].set_visible(False)
        else:
            log_tdmi_data = np.log10(tdmi_flatten)

            answer = weight_flatten[band].copy()
            answer[answer==0]=1e-7
            log_answer = np.log10(answer)
            answer_edges = np.linspace(-6, 1, num = 15)
            # average data
            log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
            for i in range(len(answer_edges)-1):
                mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
                if mask.sum() == 0:
                    log_tdmi_data_mean[i] = np.nan
                else:
                    log_tdmi_data_mean[i] = log_tdmi_data[mask].mean()
            pval = np.polyfit(answer_edges[:-1][~np.isnan(log_tdmi_data_mean)], log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], deg=1)
            ax[idx].plot(answer_edges[:-1], log_tdmi_data_mean, 'k.', markersize=15, label='TDMI mean')
            ax[idx].plot(answer_edges[:-1], np.polyval(pval, answer_edges[:-1]), 'r', label='Linear Fitting')
            ticks = [-5, -3, -1]
            labels = ['$10^{%d}$'%item for item in ticks]
            ax[idx].set_xticks(ticks)
            ax[idx].set_xticklabels(labels)
            ax[idx].set_title(f'{band:s} ($r$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval)**0.5:5.3f})')
            ax[idx].legend(fontsize=15)
            ax[idx].grid(ls='--')
        idx += 1

    # make last subfigure invisible
    ax[-1].set_visible(False)

    plt.tight_layout()
    return fig