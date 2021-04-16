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
    from utils.core import EcogTDMI
    from utils.roc import scan_auc_threshold
    from utils.plot import gen_auc_threshold_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='plot_auc_threshold',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogTDMI('data/')
    data.init_data()
    sc, fc = data.get_sc_fc('ch')
    # ==================================================

    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    aucs = {}
    for band in data.filters:
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            sc[band] = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]
        aucs[band], _ = scan_auc_threshold(fc[band], sc[band], w_thresholds)
    
    fig = gen_auc_threshold_figure(aucs, w_thresholds)

    if args.is_interarea:
        fname = f'ch_auc-threshold_mi_interarea.png'
    else:
        fname = f'ch_auc-threshold_mi.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)