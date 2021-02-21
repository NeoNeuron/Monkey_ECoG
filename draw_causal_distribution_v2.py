#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np

def load_data(path:str, band:str='raw', shuffle:bool=False):
    """Load data from files.

    Args:
        path (str): folder path of data.
        band (str, optional): name of target band. 'raw' for unfiltered. Defaults to 'raw'.
        shuffle (bool, optional): True for loading shuffled dataset. Defaults to False.

    Returns:
        np.ndarray: tdmi_data and tdmi_data_shuffle(if shuffle==True).
    """
    tdmi_data = np.load(path + 'tdmi_data.npz', allow_pickle=True)[band]
    if shuffle:
        tdmi_data_shuffle = np.load(path + 'tdmi_data_shuffle.npz', allow_pickle=True)[band]

    if shuffle:
        return tdmi_data, tdmi_data_shuffle
    else:
        return tdmi_data

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    from utils.tdmi import MI_stats
    from utils.plot import gen_causal_distribution_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'tdmi_mode': 'max',
                   'is_interarea': False,
                   'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
                   }
    parser = ArgumentParser(prog='draw_causal_distribution',
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
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]

    for band in args.filters:
        # load data for target band
        tdmi_data, tdmi_data_shuffle = load_data(args.path, band, shuffle=True)

        tdmi_data = MI_stats(tdmi_data, args.tdmi_mode)
        tdmi_data_flatten = tdmi_data[~np.eye(stride[-1], dtype=bool)]

        # setup interarea mask
        if args.is_interarea:
            tdmi_data_flatten = tdmi_data_flatten[interarea_mask]

        SI_value = tdmi_data_shuffle[~np.eye(stride[-1], dtype=bool)].mean()
        if args.tdmi_mode == 'sum':
            SI_value *= 10
        fig = gen_causal_distribution_figure(tdmi_data_flatten, 
                                             weight_flatten,
                                             SI_value)

        if args.tdmi_mode == 'sum':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
        elif args.tdmi_mode == 'max':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        if args.is_interarea:
            fname = f'channel_{band:s}_interarea_analysis_{args.tdmi_mode:s}.png'
        else:
            fname = f'channel_{band:s}_analysis_{args.tdmi_mode:s}.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)