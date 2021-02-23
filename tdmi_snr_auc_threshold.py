#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold.

import numpy as np

if __name__ == '__main__':
    import time
    import matplotlib as mpl 
    mpl.rcParams['font.size']=20
    mpl.rcParams['axes.labelsize']=25
    from utils.tdmi import MI_stats, compute_snr_matrix, get_sparsity_threshold
    from utils.roc import scan_auc_threshold
    from utils.plot import gen_auc_threshold_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'tdmi_snr_analysis/',
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

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    tdmi_data = np.load(args.path + 'tdmi_data.npz', allow_pickle=True)
    aucs = {}
    opt_threshold = {}
    snr_th = {
        'raw'        :5.0,
        'delta'      :4.3,
        'theta'      :4.3,
        'alpha'      :4,
        'beta'       :5.,
        'gamma'      :11,
        'high_gamma' :11,
    }
    for band in filter_pool:
        if band in tdmi_data.keys():
            tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
            # generate snr mask
            snr_mat = compute_snr_matrix(tdmi_data[band])
            # th_val = get_sparsity_threshold(snr_mat, p = 0.5)
            # snr_mask = snr_mat >= th_val
            snr_mask = snr_mat >= snr_th[band]
            # apply snr mask
            weight_flatten = weight[(~np.eye(stride[-1], dtype=bool))*snr_mask]
            tdmi_data_flatten = tdmi_data_band[(~np.eye(stride[-1], dtype=bool))*snr_mask]
            # setup interarea mask
            if args.is_interarea:
                interarea_mask = (weight_flatten != 1.5)
                weight_flatten = weight_flatten[interarea_mask]
                tdmi_data_flatten = tdmi_data_flatten[interarea_mask]
            
            aucs[band], opt_threshold[band] = scan_auc_threshold(tdmi_data_flatten, weight_flatten, w_thresholds)
        else:
            aucs[band], opt_threshold[band] = None, None
    
    fig = gen_auc_threshold_figure(aucs, w_thresholds)

    # save optimal threshold computed by Youden Index
    if args.is_interarea:
        np.savez(args.path + f'opt_threshold_channel_interarea_tdmi_{args.tdmi_mode:s}.npz', **opt_threshold)
    else:
        np.savez(args.path + f'opt_threshold_channel_tdmi_{args.tdmi_mode:s}.npz', **opt_threshold)

    if args.is_interarea:
        fname = f'auc-threshold_interarea_{args.tdmi_mode:s}_manual-th.png'
    else:
        fname = f'auc-threshold_{args.tdmi_mode:s}_manual-th.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)