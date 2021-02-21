#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold.

import numpy as np
import matplotlib.pyplot as plt
from draw_causal_distribution_v2 import ROC_curve, AUC, Youden_Index

def scan_auc_threshold(tdmi_data_flatten:np.ndarray, 
                       weight_flatten:np.ndarray, 
                       w_thresholds:list):
    log_tdmi_data = np.log10(tdmi_data_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # compute ROC curves for different w_threshold values
    aucs = np.zeros_like(w_thresholds)
    roc_thresholds = np.linspace(*log_tdmi_range,100)
    for iidx, threshold in enumerate(w_thresholds):
        answer = weight_flatten.copy()
        answer = (answer>threshold).astype(bool)
        fpr, tpr = ROC_curve(answer, log_tdmi_data, roc_thresholds)
        aucs[iidx] = AUC(fpr, tpr)
    opt_threshold = roc_thresholds[Youden_Index(fpr, tpr)]
    return aucs, opt_threshold

def gen_auc_threshold_figure(aucs:dict, w_thresholds:list)->plt.Figure:
    fig, ax = plt.subplots(2,4,figsize=(20,10), sharey=True)
    ax = ax.reshape((8,))
    idx = 0
    for band, auc in aucs.items():
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

if __name__ == '__main__':
    import time
    import matplotlib as mpl 
    mpl.rcParams['font.size']=20
    mpl.rcParams['axes.labelsize']=25
    from draw_causal_distribution_v2 import MI_stats
    from tdmi_scan_v2 import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                    'tdmi_mode': 'max',
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='plot_auc_threshold',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
                        type = str, choices=['max', 'sum'], 
                        help = "TDMI mode."
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    args = parser.parse_args()

    start = time.time()
    # load data
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']
    # setup interarea mask
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    tdmi_data = np.load(args.path + 'tdmi_data.npz', allow_pickle=True)
    aucs = {}
    opt_threshold = {}
    for band in filter_pool:
        tdmi_data_flatten = MI_stats(tdmi_data[band], args.tdmi_mode)
        tdmi_data_flatten = tdmi_data_flatten[~np.eye(stride[-1], dtype=bool)]
        if args.is_interarea:
            tdmi_data_flatten = tdmi_data_flatten[interarea_mask]
        
        aucs[band], opt_threshold[band] = scan_auc_threshold(tdmi_data_flatten, weight_flatten, w_thresholds)
    
    fig = gen_auc_threshold_figure(aucs, w_thresholds)

    # save optimal threshold computed by Youden Index
    if args.is_interarea:
        np.savez(args.path + f'opt_threshold_channel_interarea_tdmi_{args.tdmi_mode:s}.npz', **opt_threshold)
    else:
        np.savez(args.path + f'opt_threshold_channel_tdmi_{args.tdmi_mode:s}.npz', **opt_threshold)

    if args.is_interarea:
        fname = f'auc-threshold_interarea_{args.tdmi_mode:s}.png'
    else:
        fname = f'auc-threshold_{args.tdmi_mode:s}.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)