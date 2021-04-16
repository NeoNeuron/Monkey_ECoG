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
        fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
        fig_return_flag = True
    ax = ax.reshape((-1,))
    idx = 0
    for band, auc in aucs.items():
        if auc is None:
            ax[idx].set_visible(False)
        else:
            # plot dependence of AUC w.r.t w_threshold value
            ax[idx].semilogx(w_thresholds, auc, '-*', color=colors, label = labels)
            ax[idx].grid(ls='--')
            ax[idx].set_title(band)
        idx += 1

    [ax[i].set_ylabel('AUC') for i in (0,4)]
    [ax[i].set_xlabel('Threshold value') for i in (4,5,6)]
    ax[0].set_ylim(0.5, 0.9)

    # make last subfigure invisible
    ax[-1].set_visible(False)
    plt.tight_layout()
    if fig_return_flag:
        return fig

def gen_mi_s_figure(tdmi_data_flatten:dict, weight_flatten:dict, snr_mask:dict=None, is_log:bool=True)->plt.Figure:
    fig, ax = plt.subplots(2,4,figsize=(20,10))
    ax = ax.reshape((8,))
    idx = 0
    for band, tdmi_flatten in tdmi_data_flatten.items():
        if tdmi_flatten is None:
            ax[idx].set_visible(False)
        else:
            if is_log:
                log_tdmi_data = np.log10(tdmi_flatten)
            else:
                log_tdmi_data = tdmi_flatten.copy()
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
            pval = np.polyfit(answer_center[~np.isnan(log_tdmi_data_mean)], log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], deg=1)
            ax[idx].plot(answer_center, log_tdmi_data_mean, 'k.', markersize=15, label='TDMI mean')
            ax[idx].plot(answer_center, np.polyval(pval, answer_center), 'r', label='Linear Fitting')
            ticks = [-6, -4, -2, 0]
            labels = ['$10^{%d}$'%item for item in ticks]
            ax[idx].set_xticks(ticks)
            ax[idx].set_xticklabels(labels)
            ax[idx].set_title(f'{band:s} ($r$ = {Linear_R2(answer_center, log_tdmi_data_mean, pval)**0.5:5.3f})')
            ax[idx].legend(fontsize=15)
            ax[idx].grid(ls='--')
        idx += 1

    # make last subfigure invisible
    ax[-1].set_visible(False)

    plt.tight_layout()
    return fig

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
    fig, ax = plt.subplots(4, 2, figsize=(6, 12))
    plt_buffer_mat = np.zeros_like(roi_mask, dtype=bool)
    plt_buffer_mat[roi_mask] = weight_mask
    sorted_id = get_cluster_id(plt_buffer_mat)
    _plt_unit2(ax[0, 0], weight_mask, roi_mask, sorted_id)
    ax[0, 0].set_title('Weight Matrix')
    ax[0, 0].set_xticklabels([])
    indices = {
        'delta' : (1, 0),
        'theta' : (2, 0), 
        'alpha' : (3, 0),
        'beta'  : (1, 1),
        'gamma' : (2, 1),
        'high_gamma' : (3, 1),
        'raw'   : (0, 1)
    }
    for band, index in indices.items():
        if tdmi_mask[band] is not None:
            _plt_unit2(ax[index], tdmi_mask[band], roi_mask, sorted_id)
        ax[index].set_title(band, fontsize=16)
        ax[index].set_xticklabels([])
        ax[index].set_yticklabels([])
    [axi.invert_yaxis() for axi in ax.flatten()]
    [axi.axis('scaled') for axi in ax.flatten()]
    return fig

def plot_ppv_curves(fnames:str, figname:str):
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
    separator = [-6, -5, -4, -3, -2, -1, 0]
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    colors = ['r','royalblue','orange']
    labels = [r'PPV(th$_{fit}$)', r'PPV(th$_{gap}$)', r'PPV(th$_{roc}$)']

    for fname, color, label in zip(fnames, colors, labels):
        roc_data = np.load(fname, allow_pickle=True)
        for i, index in enumerate(indices):
            ax[index].plot(separator, 100*roc_data[:, i, -1], '-o',
                        markersize=2, markerfacecolor='None', color=color, label=label)
            # ax[index].plot(separator, 100*roc_data[:, i, -3], '-s',
            #              markerfacecolor='None', color=color, label='TPR'+label)

    for i, index in enumerate(indices):
        ax[index].plot(separator, 100*(roc_data[:,i, 0]+roc_data[:,i,2])/roc_data[:,i,0:4].sum(1),
                     '-o', markersize=2, markerfacecolor='None', color='k', label='p true')
        ax[index].grid(ls='--')
        ax[index].set_title(filters[i])

    # plot legend in the empty subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[-1, -1].legend(handles, labels, loc=1, fontsize=16)
    ax[-1, -1].axis('off')

    [ax[i, 0].set_ylabel('Percentage(%)',fontsize=16) for i in (0, 1)]
    [ax[-1, i].set_xlabel(r'$\log_{10}$(Weight thresholding)',fontsize=12) for i in [0, 1, 2]]

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

def gen_sc_fc_figure(tdmi_flatten:dict, 
                     weight_flatten:dict,
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
    # create figure canvas
    fig = plt.figure(figsize=(9,15), dpi=100)
    gs = fig.add_gridspec(nrows=4, ncols=2, 
                          left=0.10, right=0.90, top=0.96, bottom=0.05, 
                          wspace=0.25, hspace=0.35)
    ax = np.array([fig.add_subplot(i) for i in gs])

    for idx, band in enumerate(tdmi_flatten.keys()):
        if is_log:
            log_tdmi_data = np.log10(tdmi_flatten[band])
        else:
            log_tdmi_data = tdmi_flatten[band].copy()

        weight_set = np.unique(weight_flatten[band])
        log_tdmi_data_buffer = log_tdmi_data.copy()
        if snr_mask is None:
            snr_mask = np.ones_like(tdmi_flatten).astype(bool)
        log_tdmi_data_buffer[~snr_mask] = np.nan
        log_tdmi_data_mean = np.array([np.nanmean(log_tdmi_data_buffer[weight_flatten[band]==key]) for key in weight_set])
        log_tdmi_data_mean[weight_set==0]=np.nan
        weight_set[weight_set==0]=np.nan
        # pval = np.polyfit(np.log10(weight_set), log_tdmi_data_mean, deg=1)
        pval = np.polyfit(np.log10(weight_set)[~np.isnan(log_tdmi_data_mean)], log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], deg=1)
        # pval = np.polyfit(np.log10(weight_flatten+1e-6)[~np.isnan(log_tdmi_data_buffer)], log_tdmi_data_buffer[~np.isnan(log_tdmi_data_buffer)], deg=1)
        ax[idx].plot(np.log10(weight_flatten[band]+1e-6), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
        ax[idx].plot(np.log10(weight_flatten[band]+1e-6), log_tdmi_data_buffer, 'b.', label='TDMI (above SNR th)')
        ax[idx].plot(np.log10(weight_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
        ax[idx].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
        ax[idx].set_xlabel(r'$log_{10}$(Connectivity Strength)')
        weight_flatten_buffer = weight_flatten[band].copy()
        weight_flatten_buffer[weight_flatten_buffer==0] = np.nan
        ax[idx].set_title('Fitness(mean) : r = %5.3f,\n Fitness : r = %5.3f' % 
            (Linear_R2(np.log10(weight_set), log_tdmi_data_mean, pval)**0.5,Linear_R2(np.log10(weight_flatten_buffer), log_tdmi_data_buffer, pval)**0.5),
            fontsize=16
        )

    return fig

def gen_fc_rank_figure(sc, fc, is_log=True, is_interarea=False):
    fig = plt.figure(figsize=(8,15), dpi=100)
    gs = fig.add_gridspec(nrows=4, ncols=2, 
                          left=0.10, right=0.90, top=0.96, bottom=0.05, 
                          wspace=0.36, hspace=0.30)
    ax = np.array([fig.add_subplot(i) for i in gs])
    axt = []

    for idx, band in enumerate(sc.keys()):
        axt.append(ax[idx].twinx())
        if fc[band] is not None:
            # setup interarea mask
            if is_interarea:
                interarea_mask = (sc[band] != 1.5)
                sc[band] = sc[band][interarea_mask]
                fc[band] = fc[band][interarea_mask]

            if is_log:
                ax[idx].plot(np.log10(np.sort(fc[band])), np.arange(fc[band].shape[0]), '.', ms=0.1)
                gap_th_val, gap_th_label = find_gap_threshold(np.log10(fc[band]))
                ax[idx].axvline(gap_th_val, color='r', label=gap_th_label)
                gap_th_val = 10**gap_th_val
                axt[idx].hist(np.log10(fc[band][sc[band]>0]),color='orange', alpha=.5, bins=100, label='SC Present')
                axt[idx].hist(np.log10(fc[band][sc[band]==0]), color='navy', alpha=.5, bins=100, label='SC Absent')
            else:
                ax[idx].plot((np.sort(fc[band])), np.arange(fc[band].shape[0]), '.', ms=0.1)
                gap_th_val, gap_th_label = find_gap_threshold((fc[band]))
                ax[idx].axvline(gap_th_val, color='r', label=gap_th_label)
                axt[idx].hist((fc[band][sc[band]>0]),color='orange', alpha=.5, bins=100, label='SC Present')
                axt[idx].hist((fc[band][sc[band]==0]), color='navy', alpha=.5, bins=100, label='SC Absent')

            # styling
            ax[idx].legend(fontsize=10, loc=5)
            ax[idx].yaxis.get_major_formatter().set_powerlimits((0,1))
            ax[idx].set_title(band)
            ax[idx].text(
                0.05, 0.95, 
                f'PPV:{np.sum(fc[band][sc[band]>0]>gap_th_val)*100./np.sum(fc[band]>gap_th_val):4.1f} %',
                fontsize=14, transform=ax[idx].transAxes, 
                verticalalignment='top', horizontalalignment='left'
            )

    [ax[i].set_ylabel('Ranked MI index') for i in (0,2,4,6)]
    [ax[i].set_xlabel('MI value') for i in (5,6)]
    [axt[i].set_ylabel('Counts') for i in (1,3,5)]
    handles, labels = axt[0].get_legend_handles_labels()
    ax[-1].legend(handles, labels, fontsize=16, loc=2)
    ax[-1].axis('off')
    return fig

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
