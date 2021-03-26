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
    from utils.tdmi import MI_stats
    from utils.roc import scan_auc_threshold
    from utils.plot import gen_auc_threshold_figure
    from utils.utils import print_log
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
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']
    # setup interarea mask
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
    aucs = {}
    for band in filter_pool:
        if band in tdmi_data.keys():
            tdmi_data_flatten = MI_stats(tdmi_data[band], args.tdmi_mode)
            tdmi_data_flatten = tdmi_data_flatten[~np.eye(stride[-1], dtype=bool)]
            if args.is_interarea:
                tdmi_data_flatten = tdmi_data_flatten[interarea_mask]
            
            aucs[band], _ = scan_auc_threshold(tdmi_data_flatten, weight_flatten, w_thresholds)
        else:
            aucs[band] = None
    
    fig = gen_auc_threshold_figure(aucs, w_thresholds)

    if args.is_interarea:
        fname = f'auc-threshold_interarea_{args.tdmi_mode:s}.png'
    else:
        fname = f'auc-threshold_{args.tdmi_mode:s}.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)