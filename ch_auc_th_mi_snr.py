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
    from utils.roc import scan_auc_threshold
    from utils.plot import gen_auc_threshold_figure
    from utils.utils import print_log
    from utils.core import EcogTDMI
    import pickle
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'tdmi_snr_analysis/',
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='plot_auc_threshold',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument(
        'is_interarea', 
        default=arg_default['is_interarea'], nargs='?', 
        type=bool, 
        help = "inter-area flag."
    )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    # no_snr_mask
    data_no_snr = EcogTDMI('data/')
    data_no_snr.init_data()
    sc_no_snr, fc_no_snr = data_no_snr.get_sc_fc('ch')

    data = EcogTDMI('data/')
    data.init_data(args.path)
    sc, fc = data.get_sc_fc('ch')
    # ==================================================

    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    aucs = {}
    aucs_no_snr = {}
            
    if args.is_interarea:
        interarea_mask = (sc[data.filters[0]] != 1.5)
    for band in data.filters:
        # setup interarea mask
        if args.is_interarea:
            sc[band] = sc[band][interarea_mask]
            fc_no_snr[band] = sc_no_snr[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]
        aucs[band], _ = scan_auc_threshold(fc[band], sc[band], w_thresholds)
        aucs_no_snr[band], _ = scan_auc_threshold(fc_no_snr[band], sc_no_snr[band], w_thresholds)
    
    fig = gen_auc_threshold_figure(aucs_no_snr, w_thresholds, labels='No SNR mask')
    gen_auc_threshold_figure(aucs, w_thresholds, ax=np.array(fig.get_axes()), colors='orange', labels='SNR mask')
    [axi.legend() for axi in fig.get_axes()[:-1]]

    if args.is_interarea:
        fname = f'auc-threshold_interarea_manual-th.png'
    else:
        fname = f'auc-threshold_manual-th.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    if args.is_interarea:
        with open(args.path+f'aucs_interarea.pkl', 'wb') as f:
            pickle.dump(aucs_no_snr, f)
            pickle.dump(aucs, f)
        print_log(f'Figure save to {args.path:s}aucs_interarea.pkl', start)
    else:
        with open(args.path+f'aucs.pkl', 'wb') as f:
            pickle.dump(aucs_no_snr, f)
            pickle.dump(aucs, f)
        print_log(f'Figure save to {args.path:s}aucs.pkl', start)