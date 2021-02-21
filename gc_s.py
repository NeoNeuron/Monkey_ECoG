#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import numpy as np

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt 
    plt.rcParams['font.size']=15
    plt.rcParams['axes.labelsize'] = 15
    import matplotlib.pyplot as plt
    from plot_mi_s import gen_mi_s_figure
    from tdmi_scan_v2 import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                    'order': 6,
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='gc_s',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('order', default=arg_default['order'], nargs='?',
                        type = int,
                        help = "order of regression model in GC."
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    # setup interarea mask
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]
    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

    gc_data = np.load(args.path + f'gc_order_{args.order:d}.npz', allow_pickle=True)

    gc_data_flatten = {}
    for band in filter_pool:
        gc_data_flatten[band] = gc_data[band][~np.eye(stride[-1], dtype=bool)]
        gc_data_flatten[band][gc_data_flatten[band]<=0] = 1e-5
        if args.is_interarea:
            gc_data_flatten[band] = gc_data_flatten[band][interarea_mask]

    fig = gen_mi_s_figure(gc_data_flatten, weight_flatten)

    # edit axis labels
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
        labels = [item.replace('TDMI', 'GC') for item in labels]
        ax.legend(handles, labels)
    [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(GC\right)$') for i in (0,4)]
    [fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
    plt.tight_layout()

    if args.is_interarea:
        fname = f'gc-s_interarea_{args.order:d}.png'
    else:
        fname = f'gc-s_{args.order:d}.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)