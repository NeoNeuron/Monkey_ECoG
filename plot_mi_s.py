#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size']=15
    plt.rcParams['axes.labelsize'] = 15
    from utils.tdmi import MI_stats
    from utils.utils import print_log
    from utils.plot import gen_mi_s_figure
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                    'tdmi_mode': 'max',
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='plot_mi_s',
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
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']
    # setup interarea mask
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]
    tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

    weight_flatten = {band:weight_flatten for band in filter_pool}
    tdmi_data_flatten = {}
    for band in filter_pool:
        if band in tdmi_data.keys():
            tdmi_data_flatten[band] = MI_stats(tdmi_data[band], args.tdmi_mode)
            tdmi_data_flatten[band] = tdmi_data_flatten[band][~np.eye(stride[-1], dtype=bool)]
            if args.is_interarea:
                tdmi_data_flatten[band] = tdmi_data_flatten[band][interarea_mask]
        else:
            tdmi_data_flatten[band] = None

    fig = gen_mi_s_figure(tdmi_data_flatten, weight_flatten)

    # edit axis labels
    if args.tdmi_mode == 'sum':
        [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$') for i in (0,4)]
    elif args.tdmi_mode == 'max':
        [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$') for i in (0,4)]
    [fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
    plt.tight_layout()

    if args.is_interarea:
        fname = f'mi-s_interarea_{args.tdmi_mode:s}.png'
    else:
        fname = f'mi-s_{args.tdmi_mode:s}.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)