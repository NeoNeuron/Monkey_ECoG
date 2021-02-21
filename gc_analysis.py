#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np

def load_data(path:str, band:str='raw', order:int=10, shuffle:bool=False):
    """Load data from files.

    Args:
        path (str): folder path of data.
        band (str, optional): name of target band. 'raw' for unfiltered. Defaults to 'raw'.
        order (int, optional): order of regression.
        shuffle (bool, optional): True for loading shuffled dataset. Defaults to False.

    Returns:
        np.ndarray: gc_data and gc_data_shuffle(if shuffle==True).
    """
    gc_data = np.load(path + f'gc_order_{order:d}.npz', allow_pickle=True)[band]
    if shuffle:
        gc_data_shuffle = np.load(path + f'gc_shuffled_order_{order:d}.npz', allow_pickle=True)[band]

    if shuffle:
        return gc_data, gc_data_shuffle
    else:
        return gc_data

if __name__ == '__main__':
    import time
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import gen_causal_distribution_figure
    from tdmi_scan_v2 import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'order': 6,
                   'is_interarea': False,
                   'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
                   }
    parser = ArgumentParser(prog='gc_analysis',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('order', default=arg_default['order'], nargs='?',
                        type = int, 
                        help = "order of regressio model in GC"
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    parser.add_argument('--filters', default=arg_default['filters'], nargs='*', 
                        type=str, 
                        help = "list of filtering band."
                        )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    # prepare weight_flatten
    weight = data_package['weight']
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    # setup interarea mask
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]

    for band in args.filters:
        # load data for target band
        gc_data, gc_data_shuffle = load_data(args.path, band, args.order, shuffle=True)
        # gc_data = load_data(args.path, band, args.order, shuffle=False)
        gc_data_flatten = gc_data[~np.eye(stride[-1], dtype=bool)]
        gc_data_flatten[gc_data_flatten<=0] = 1e-5


        if args.is_interarea:
            gc_data_flatten = gc_data_flatten[interarea_mask]

        SI_value = gc_data_shuffle[~np.eye(stride[-1], dtype=bool)]
        SI_value[SI_value<=0] = 0
        SI_value = SI_value.mean()
        fig = gen_causal_distribution_figure(gc_data_flatten, 
                                             weight_flatten,
                                             SI_value)

        fig.get_axes()[4].set_ylabel(r'$log_{10}\left(GC\right)$')
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        labels = [item.replace('TDMI', 'GC') for item in labels]
        fig.get_axes()[4].legend(handles, labels)
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        if args.is_interarea:
            fname = f'channel_{band:s}_gc_interarea_order_{args.order:d}.png'
        else:
            fname = f'channel_{band:s}_gc_order_{args.order:d}.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)