#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    from utils.utils import CG, print_log
    from utils.plot import gen_causal_distribution_figure
    from gc_analysis import load_data
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'order': 6,
                   'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
                   }
    parser = ArgumentParser(prog='GC CG_causal_distribution',
                            description = "Generate figure for coarse-grain analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('order', default=arg_default['order'], nargs='?',
                        type = int, 
                        help = "regression order in GC."
                        )
    parser.add_argument('--filters', default=arg_default['filters'], nargs='*', 
                        type=str, 
                        help = "list of filtering band."
                        )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    n_region = multiplicity.shape[0]

    # create adj_weight_flatten by excluding 
    #   auto-gc in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    for band in args.filters:
        # load shuffled gc data for target band
        gc_data, gc_data_shuffle = load_data(args.path, band, args.order, shuffle=True)
        gc_data_cg = CG(gc_data, stride)

        gc_data_flatten = gc_data_cg[cg_mask]
        gc_data_flatten[gc_data_flatten<=0] = 1e-5

        SI_value = gc_data_shuffle[~np.eye(stride[-1], dtype=bool)]
        SI_value[SI_value<=0] = 0
        SI_value = SI_value.mean()

        fig = gen_causal_distribution_figure(gc_data_flatten, 
                                             adj_weight_flatten,
                                             SI_value,
                                             )
        fig.get_axes()[4].set_ylabel(r'$log_{10}\left(GC\right)$')
        fig.get_axes()[4].get_lines()[1].set_color((0,0,0,0))
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        labels = [item.replace('TDMI', 'GC') for item in labels]
        fig.get_axes()[4].legend((handles[0],handles[2]), (labels[0], labels[2]))
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        fname = f'cg_{band:s}_gc_order_{args.order:d}.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)