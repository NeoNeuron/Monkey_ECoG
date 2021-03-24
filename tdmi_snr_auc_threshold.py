#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold.


if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib as mpl 
    mpl.rcParams['font.size']=20
    mpl.rcParams['axes.labelsize']=25
    from utils.tdmi import MI_stats, compute_snr_matrix, compute_noise_matrix
    from utils.roc import scan_auc_threshold
    from utils.plot import gen_auc_threshold_figure
    from utils.utils import print_log
    import pickle
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
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    weight = data_package['weight']
    off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    w_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
    aucs = {}
    aucs_no_snr = {}
    opt_threshold = {}
    opt_threshold_no_snr = {}
    with open(args.path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)
    for band in filter_pool:
        if band in tdmi_data.keys():
            tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
            tdmi_data_flatten_no_snr = tdmi_data_band[off_diag_mask]
            # generate snr mask
            snr_mat = compute_snr_matrix(tdmi_data[band])
            noise_mat = compute_noise_matrix(tdmi_data[band])
            snr_mask = snr_mat >= snr_th[band]
            # apply snr mask
            tdmi_data_band[~snr_mask] = noise_mat[~snr_mask]
            tdmi_data_flatten = tdmi_data_band[off_diag_mask]
            # flatten weight matrix
            weight_flatten = weight[off_diag_mask]
            # setup interarea mask
            if args.is_interarea:
                interarea_mask = (weight_flatten != 1.5)
                weight_flatten = weight_flatten[interarea_mask]
                tdmi_data_flatten_no_snr = tdmi_data_flatten[interarea_mask]
                tdmi_data_flatten = tdmi_data_flatten[interarea_mask]
            
            aucs_no_snr[band], opt_threshold_no_snr[band] = scan_auc_threshold(tdmi_data_flatten_no_snr, weight_flatten, w_thresholds)
            aucs[band], opt_threshold[band] = scan_auc_threshold(tdmi_data_flatten, weight_flatten, w_thresholds)
        else:
            aucs_no_snr[band], opt_threshold_no_snr[band] = None, None
            aucs[band], opt_threshold[band] = None, None
    
    fig = gen_auc_threshold_figure(aucs_no_snr, w_thresholds, labels='No SNR mask')
    gen_auc_threshold_figure(aucs, w_thresholds, ax=np.array(fig.get_axes()), colors='orange', labels='SNR mask')
    [axi.legend() for axi in fig.get_axes()[:-1]]

    # save optimal threshold computed by Youden Index
    if args.is_interarea:
        with open(args.path+f'opt_threshold_channel_interarea_tdmi_{args.tdmi_mode:s}.pkl', 'wb') as f:
            snr_th = pickle.dump(opt_threshold, f)
    else:
        with open(args.path+f'opt_threshold_channel_tdmi_{args.tdmi_mode:s}.pkl', 'wb') as f:
            snr_th = pickle.dump(opt_threshold, f)

    if args.is_interarea:
        fname = f'auc-threshold_interarea_{args.tdmi_mode:s}_manual-th.png'
    else:
        fname = f'auc-threshold_{args.tdmi_mode:s}_manual-th.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)