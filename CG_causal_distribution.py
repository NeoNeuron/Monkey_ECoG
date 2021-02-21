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
    from utils.tdmi import Extract_MI_CG
    from draw_causal_distribution_v2 import load_data
    from utils.plot import gen_causal_distribution_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'tdmi_mode': 'max',
                   'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
                   }
    parser = ArgumentParser(prog='CG_causal_distribution',
                            description = "Generate figure for coarse-grain analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
                        type = str, choices=['max', 'sum'], 
                        help = "TDMI mode."
                        )
    parser.add_argument('--filters', default=arg_default['filters'], nargs='*', 
                        type=str, 
                        help = "list of filtering band."
                        )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    n_region = multiplicity.shape[0]

    # create adj_weight_flatten by excluding 
    #   auto-tdmi in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    for band in args.filters:
        # load shuffled tdmi data for target band
        tdmi_data, tdmi_data_shuffle = load_data(args.path, band, shuffle=True)
        tdmi_data_cg = Extract_MI_CG(tdmi_data, args.tdmi_mode, stride)

        tdmi_data_flatten = tdmi_data_cg[cg_mask]

        SI_value = tdmi_data_shuffle[~np.eye(stride[-1], dtype=bool)].mean()
        if args.tdmi_mode == 'sum':
            SI_value *= 10

        fig = gen_causal_distribution_figure(tdmi_data_flatten, 
                                             adj_weight_flatten,
                                             SI_value,
                                             )

        if args.tdmi_mode == 'sum':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
        elif args.tdmi_mode == 'max':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        fig.get_axes()[4].get_lines()[1].set_color((0,0,0,0))
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        fig.get_axes()[4].legend((handles[0],handles[2]), (labels[0], labels[2]))
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        fname = f'cg_{band:s}_analysis_{args.tdmi_mode:s}.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)