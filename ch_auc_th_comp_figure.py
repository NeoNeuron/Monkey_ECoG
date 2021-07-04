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
    from fcpy.plot import gen_auc_threshold_figure
    from fcpy.utils import print_log
    import pickle
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'tdmi_snr_analysis/'}
    parser = ArgumentParser(prog='plot_auc_threshold',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    args = parser.parse_args()

    start = time.time()
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    with open(args.path + 'ch_aucs.pkl', 'rb') as f:
        aucs_tdmi = pickle.load(f)
    with open(args.path + 'ch_aucs_gc_order_6.pkl', 'rb') as f:
        aucs_gc = pickle.load(f)
    
    fig = gen_auc_threshold_figure(aucs_tdmi, w_thresholds, labels='TDMI')
    gen_auc_threshold_figure(aucs_gc, w_thresholds, ax=np.array(fig.get_axes()), colors='orange', labels='GC')
    [axi.legend() for axi in fig.get_axes()[:-1]]
    fig.get_axes()[0].set_ylim(0.45,0.85)

    fname = f'ch_auc-threshold_summary.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)