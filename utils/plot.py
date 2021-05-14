import numpy as np
import matplotlib.pyplot as plt
from .roc import ROC_curve, AUC, Youden_Index
from .utils import Linear_R2
from .cluster import get_cluster_id, get_sorted_mat
from .binary_threshold import find_gap_threshold

def gen_causal_distribution_figure(tdmi_flatten:np.ndarray, 
                                   weight_flatten:np.ndarray,
                                   tdmi_threshold:float, 
                                   snr_mask:np.ndarray=None,
                                   is_log:bool=True,
                                   )->plt.Figure:
    """Generated figure for analysis of causal distributions.

    Args:
        tdmi_flatten (np.ndarray): flattened data for target tdmi statistics.
        weight_flatten (np.ndarray): flattened data for true connectome.
        tdmi_threshold (float): significance value of tdmi statistics.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    if is_log:
        log_tdmi_data = np.log10(tdmi_flatten)
    else:
        log_tdmi_data = tdmi_flatten.copy()
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # calculate histogram
    (counts, edges) = np.histogram(log_tdmi_data, bins=100, density=True)

    # create figure canvas
    fig, ax = plt.subplots(2,4,figsize=(20,10))

    ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
    # ax[0,0].axvline(np.log10(tdmi_threshold), color='cyan', label='SI')

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
    ax[0,0].legend(fontsize=13, loc=1)

    weight_set = np.unique(weight_flatten)
    log_tdmi_data_buffer = log_tdmi_data.copy()
    if snr_mask is None:
        snr_mask = np.ones_like(tdmi_flatten).astype(bool)
    log_tdmi_data_buffer[~snr_mask] = np.nan
    log_tdmi_data_mean = np.array([np.nanmean(log_tdmi_data_buffer[weight_flatten==key]) for key in weight_set])
    log_tdmi_data_mean[weight_set==0]=np.nan
    weight_set[weight_set==0]=np.nan
    # pval = np.polyfit(np.log10(weight_set), log_tdmi_data_mean, deg=1)
    pval = np.polyfit(np.log10(weight_set)[~np.isnan(log_tdmi_data_mean)], log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], deg=1)
    # pval = np.polyfit(np.log10(weight_flatten+1e-6)[~np.isnan(log_tdmi_data_buffer)], log_tdmi_data_buffer[~np.isnan(log_tdmi_data_buffer)], deg=1)
    ax[1,0].plot(np.log10(weight_flatten+1e-6), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[1,0].plot(np.log10(weight_flatten+1e-6), log_tdmi_data_buffer, 'b.', label='TDMI (above SNR th)')
    ax[1,0].plot(np.log10(weight_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
    ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
    ax[1,0].set_xlabel(r'$log_{10}$(Connectivity Strength)')
    weight_flatten_buffer = weight_flatten.copy()
    weight_flatten_buffer[weight_flatten_buffer==0] = np.nan
    ax[1,0].set_title('Fitness(mean) : r = %5.3f,\n Fitness : r = %5.3f' % 
        (Linear_R2(np.log10(weight_set), log_tdmi_data_mean, pval)**0.5,Linear_R2(np.log10(weight_flatten_buffer), log_tdmi_data_buffer, pval)**0.5),
        fontsize=16
    )
    ax[1,0].legend(fontsize=15)

    # Draw ROC curves
    threshold_options = [1e-1, 5e-3, 1e-4]
    opt_threshold = np.zeros(len(threshold_options))
    for idx, threshold in enumerate(threshold_options):
        answer = weight_flatten.copy()
        ax[0,idx+1].semilogy(np.sort(answer), '.', ms=.1)
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

    ax[1,0].axhline(opt_threshold.mean(), color='orange', label='opt_threshold')
    ax[0,0].axvline(opt_threshold.mean(), color='orange', label='opt_threshold')
    ax[0,0].legend(fontsize=13, loc=1)

    plt.tight_layout()
    return fig

def gen_auc_threshold_figure(aucs:dict, w_thresholds:list, 
                             ax:np.ndarray=None, colors:str='navy',labels:str=None,)->plt.Figure:
    fig_return_flag = False
    if ax is None:
        fig, ax = plt.subplots(2,5,figsize=(24,10), sharey=True)
        fig_return_flag = True
    ax = ax.flatten()
    for i, (band, auc) in enumerate(aucs.items()):
        ax[i] = gen_th_auc_figure_single(ax[i], w_thresholds, auc, band, colors, labels)

    # make last subfigure invisible
    ax[-1].set_visible(False)
    plt.tight_layout()
    if fig_return_flag:
        return fig
    else:
        return ax

def gen_th_auc_figure_single(ax, w_thresholds:list, auc:np.ndarray, band:str,
    colors:str='navy',labels:str=None,)->plt.Figure:
    if auc is None:
        ax.set_visible(False)
    else:
        # plot dependence of AUC w.r.t w_threshold value
        ax.semilogx(w_thresholds, auc, '-*', color=colors, label = labels)
        ax.grid(ls='--')
        ax.set_title(band)

        ax.set_ylabel('AUC')
        ax.set_xlabel('Threshold value')
        ax.set_ylim(0.5, 0.9)
    return ax 

def gen_mi_s_figure(tdmi_data_flatten:dict, weight_flatten:dict, snr_mask:dict=None, is_log:bool=True)->plt.Figure:
    fig, ax = plt.subplots(2,5,figsize=(24,10))
    ax = ax.flatten()
    for i, band in enumerate(tdmi_data_flatten.keys()):
        if snr_mask is None:
            ax[i] = gen_sc_fc_figure_cg_single(ax[i], tdmi_data_flatten[band], weight_flatten[band], band, None, is_log)
        else:
            ax[i] = gen_sc_fc_figure_cg_single(ax[i], tdmi_data_flatten[band], weight_flatten[band], band, snr_mask[band], is_log)
    # make last subfigure invisible
    ax[-1].set_visible(False)
    plt.tight_layout()
    return fig

def gen_sc_fc_figure_cg_single(ax,
    fc:np.ndarray, 
    sc:np.ndarray, 
    band:str,
    snr_mask:np.ndarray=None, 
    is_log:bool=True,
)->plt.Figure:
    if fc is None:
        ax.set_visible(False)
    else:
        if is_log:
            log_fc = np.log10(fc)
        else:
            log_fc = fc.copy()
        if snr_mask is not None:
            log_fc[~snr_mask] = np.nan

        answer = sc.copy()
        answer[answer==0]=1e-7
        log_answer = np.log10(answer)
        answer_edges = np.linspace(-6, 1, num = 50)
        answer_center = (answer_edges[1:] + answer_edges[:-1])/2
        # average data
        log_fc_mean = np.zeros(len(answer_edges)-1)
        for i in range(len(answer_edges)-1):
            mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
            if mask.sum() == 0:
                log_fc_mean[i] = np.nan
            else:
                log_fc_mean[i] = np.nanmean(log_fc[mask])
        pval = np.polyfit(answer_center[~np.isnan(log_fc_mean)], log_fc_mean[~np.isnan(log_fc_mean)], deg=1)
        ax.plot(answer_center, log_fc_mean, 'k.', markersize=15, label='TDMI mean')
        ax.plot(answer_center, np.polyval(pval, answer_center), 'r', label='Linear Fitting')
        ticks = [-6, -4, -2, 0]
        labels = ['$10^{%d}$'%item for item in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_title(f'{band:s} ($r$ = {Linear_R2(answer_center, log_fc_mean, pval)**0.5:5.3f})')
        ax.legend(fontsize=15)
        ax.grid(ls='--')
    return ax

def gen_binary_recon_figure(tdmi_mask:dict, weight_mask:np.ndarray, roi_mask:np.ndarray):
    """Generate figure for all bands with respect to same weight_mask.

    Args:
        tdmi_mask (dict): dictionary for all bands of binary FCs.
        weight_mask (np.ndarray): binary SCs.
        roi_mask (np.ndarray): coordinate mask for SC and FCs in original adj matrix.
    """
    def _plt_unit2(axi, mat, mask, sorted_id=None):
        mat_buffer = np.zeros_like(mask, dtype=bool)
        mat_buffer[mask] = mat
        if sorted_id is not None:
            buffer = get_sorted_mat(mat_buffer, sorted_id)
            axi.pcolormesh(buffer, cmap=plt.cm.gray)
        else:
            axi.pcolormesh(mat_buffer, cmap=plt.cm.gray)

    # plot figure
    fig, ax = plt.subplots(5, 2, figsize=(6, 15))
    ax = ax.flatten()
    plt_buffer_mat = np.zeros_like(roi_mask, dtype=bool)
    plt_buffer_mat[roi_mask] = weight_mask
    sorted_id = get_cluster_id(plt_buffer_mat)
    _plt_unit2(ax[0], weight_mask, roi_mask, sorted_id)
    ax[0].set_title('Weight Matrix')
    ax[0].set_xticklabels([])
    index_order = [
        'raw','delta','theta','alpha','beta',
        'gamma','high_gamma','sub_delta','above_delta',
    ]
    for idx, band in enumerate(index_order):
        if tdmi_mask[band] is not None:
            _plt_unit2(ax[idx+1], tdmi_mask[band], roi_mask, sorted_id)
        ax[idx+1].set_title(band, fontsize=16)
        ax[idx+1].set_xticklabels([])
        ax[idx+1].set_yticklabels([])
    [axi.invert_yaxis() for axi in ax]
    [axi.axis('scaled') for axi in ax]
    return fig

def plot_ppv_curve(ax, roc_data, colors, labels, band=None):
    separator = [-6, -5, -4, -3, -2, -1, 0]
    for roc, color, label in zip(roc_data, colors, labels):
        ax.plot(separator, 100*roc[:, -1], '-o',
                markersize=2, markerfacecolor='None', color=color, label=label)
        # ax.plot(separator, 100*roc[:, i, -3], '-s',
        #         markerfacecolor='None', color=color, label='TPR'+label)

    ax.plot(separator, 100*(roc[:, 0]+roc[:,2])/roc[:,0:4].sum(1),
            '-o', markersize=2, markerfacecolor='None', color='gray', ls='--', label='p true')
    ax.grid(ls='--')
    ax.set_ylim(0,100)
    ax.set_ylabel('Percentage(%)')
    ax.set_xlabel(r'$\log_{10}$(SCs Thresholding)')
    if band is not None:
        ax.set_title(band)
    return ax

def plot_ppv_curves(fnames:str, figname:str):
    fig, ax = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
    separator = [-6, -5, -4, -3, -2, -1, 0]
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw', 'sub_delta', 'above_delta']
    ax = ax.flatten()
    colors = ['r','royalblue','orange']
    labels = [r'PPV(th$_{fit}$)', r'PPV(th$_{gap}$)', r'PPV(th$_{roc}$)']

    for fname, color, label in zip(fnames, colors, labels):
        roc_data = np.load(fname, allow_pickle=True)
        for i in range(len(filters)):
            ax[i].plot(separator, 100*roc_data[:, i, -1], '-o',
                        markersize=2, markerfacecolor='None', color=color, label=label)
            # ax[i].plot(separator, 100*roc_data[:, i, -3], '-s',
            #              markerfacecolor='None', color=color, label='TPR'+label)

    for i in range(len(filters)):
        ax[i].plot(separator, 100*(roc_data[:,i, 0]+roc_data[:,i,2])/roc_data[:,i,0:4].sum(1),
                 '-o', markersize=2, markerfacecolor='None', color='k', label='p true')
        ax[i].grid(ls='--')
        ax[i].set_title(filters[i])

    # plot legend in the empty subplot
    handles, labels = ax[0].get_legend_handles_labels()
    ax[-1].legend(handles, labels, loc=1, fontsize=16)
    ax[-1].axis('off')

    [ax[i].set_ylabel('Percentage(%)',fontsize=16) for i in (0, 1)]
    [ax[i].set_xlabel(r'$\log_{10}$(Weight thresholding)',fontsize=12) for i in [0, 1, 2]]

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

def gen_sc_fc_figure(tdmi_flatten:dict, 
                     weight_flatten:dict,
                     tdmi_threshold:float, 
                     snr_mask:dict=None,
                     is_log:bool=True,
                     )->plt.Figure:
    """Generated figure for analysis of causal distributions.

    Args:
        tdmi_flatten (np.ndarray): flattened data for target tdmi statistics.
        weight_flatten (np.ndarray): flattened data for true connectome.
        tdmi_threshold (float): significance value of tdmi statistics.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    # create figure canvas
    fig = plt.figure(figsize=(15,15), dpi=100)
    gs = fig.add_gridspec(nrows=3, ncols=3, 
                          left=0.05, right=0.95, top=0.96, bottom=0.05, 
                          wspace=0.25, hspace=0.35)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, band in enumerate(tdmi_flatten.keys()):
        if snr_mask is None:
            ax[i] = gen_sc_fc_figure_single(ax[i], tdmi_flatten[band], weight_flatten[band], band, None, is_log)
        else:
            ax[i] = gen_sc_fc_figure_single(ax[i], tdmi_flatten[band], weight_flatten[band], band, snr_mask[band], is_log)
    return fig

def gen_sc_fc_figure_single(ax,
    fc:np.ndarray, sc:np.ndarray, band:str,
    snr_mask:np.ndarray=None, is_log:bool=True,
    **kwargs
)->plt.Figure:

    if snr_mask is None:
        snr_mask = np.ones_like(fc).astype(bool)
    if is_log:
        log_fc = np.log10(fc)
        ax.set_ylabel(r'$log_{10}$(TDMI)')
    else:
        log_fc = fc.copy()
        ax.set_ylabel(r'TDMI')

    weight_set = np.unique(sc)
    log_fc_buffer = log_fc.copy()
    log_fc_buffer[~snr_mask] = np.nan
    log_fc_mean = np.array([np.nanmean(log_fc_buffer[sc==key]) for key in weight_set])
    log_fc_mean[weight_set==0]=np.nan
    weight_set[weight_set==0]=np.nan
    # pval = np.polyfit(np.log10(weight_set), log_fc_mean, deg=1)
    pval = np.polyfit(np.log10(weight_set)[~np.isnan(log_fc_mean)], log_fc_mean[~np.isnan(log_fc_mean)], deg=1)
    # pval = np.polyfit(np.log10(sc+1e-6)[~np.isnan(log_fc_buffer)], log_fc_buffer[~np.isnan(log_fc_buffer)], deg=1)
    ax.plot(np.log10(sc+1e-6), log_fc.flatten(), 'k.', label='TDMI samples')
    ax.plot(np.log10(sc+1e-6), log_fc_buffer, 'b.', label='TDMI (above SNR th)')
    ax.plot(np.log10(weight_set), log_fc_mean, 'm.', label='TDMI mean')
    ax.plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
    ax.set_xlabel(r'$log_{10}$(Connectivity Strength)')
    sc_buffer = sc.copy()
    sc_buffer[sc_buffer==0] = np.nan
    ax.set_title(band+'\n r(mean) = %5.3f, r = %5.3f' % 
        (Linear_R2(np.log10(weight_set), log_fc_mean, pval)**0.5,Linear_R2(np.log10(sc_buffer), log_fc_buffer, pval)**0.5),
        fontsize=16
    )
    return ax

def gen_fc_rank_figure(sc:dict, fc:dict, snr_mask:dict=None, is_log=True, is_interarea=False):
    fig = plt.figure(figsize=(8,15), dpi=100)
    gs = fig.add_gridspec(nrows=4, ncols=2, 
                          left=0.10, right=0.90, top=0.96, bottom=0.05, 
                          wspace=0.36, hspace=0.30)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, band in enumerate(fc.keys()):
        if fc[band] is not None:
            if snr_mask is None:
                ax[i] = gen_fc_rank_figure_single(ax[i], sc[band], fc[band], band, None, is_log, is_interarea)
            else:
                ax[i] = gen_fc_rank_figure_single(ax[i], sc[band], fc[band], band, snr_mask[band], is_log, is_interarea)

    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_ylabel('Counts') for i in (0,2,4,6)]
    [ax[i].set_xlabel('MI value') for i in (5,6)]
    handles, labels = ax[0].get_legend_handles_labels()
    ax[-1].legend(handles, labels, fontsize=16, loc=2)
    ax[-1].axis('off')
    return fig

def gen_fc_rank_figure_single(ax, sc:np.ndarray, fc:np.ndarray, band:str, snr_mask:np.ndarray=None, is_log=True, is_interarea=False):
    if snr_mask is None:
        snr_mask = np.ones_like(sc, dtype=bool)
    # setup interarea mask
    if is_interarea:
        interarea_mask = (sc != 1.5)
        sc = sc[interarea_mask]
        fc = fc[interarea_mask]

    if is_log:
        gap_th_val, gap_th_label = find_gap_threshold(np.log10(fc))
        ax.axvline(gap_th_val, color='r', label=gap_th_label)
        gap_th_val = 10**gap_th_val
        ax.hist(np.log10(fc[sc>0]),color='orange', alpha=.5, bins=100, label='SC Present')
        ax.hist(np.log10(fc[sc==0]), color='navy', alpha=.5, bins=100, label='SC Absent')
    else:
        gap_th_val, gap_th_label = find_gap_threshold((fc))
        ax.axvline(gap_th_val, color='r', label=gap_th_label)
        ax.hist((fc[sc>0]),color='orange', alpha=.5, bins=100, label='SC Present')
        ax.hist((fc[sc==0]), color='navy', alpha=.5, bins=100, label='SC Absent')

    # styling
    ax.legend(fontsize=10, loc=5)
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    ax.set_title(band)
    ax.text(
        0.05, 0.95, 
        f'PPV:{np.sum(fc[(sc>0)*snr_mask]>gap_th_val)*100./np.sum(fc[snr_mask]>gap_th_val):4.1f} %',
        fontsize=14, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='left'
    )
    ax.set_ylabel('Counts')
    ax.set_xlabel('MI value')
    return ax


def gen_binary_recon_figures(fname:str, sc_mask:list, fc_mask:dict, roi_mask):
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 0.1
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    # reshape fc_mask
    for key, item in fc_mask.items():
        if item is not None:
            if len(item.shape) == 1:
                fc_mask[key] = np.tile(item, (len(w_thresholds),1))
    fc_mask_buffer = [{}]*len(w_thresholds)
    for band, item in fc_mask.items():
        if item is not None:
            for idx in range(len(w_thresholds)):
                fc_mask_buffer[idx][band] = item[idx]
        else:
            for idx in range(len(w_thresholds)):
                fc_mask_buffer[idx][band] = None

    for idx in range(len(w_thresholds)):
        fig = gen_binary_recon_figure(fc_mask_buffer[idx], sc_mask[idx], roi_mask)
        plt.tight_layout()
        fig.savefig(fname.replace('.pkl', f'_{idx:d}.png'))
        plt.close()
