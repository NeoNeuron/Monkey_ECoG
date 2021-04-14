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
    from utils.core import EcogGC
    from utils.utils import print_log
    from utils.plot import gen_causal_distribution_figure
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
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogGC()
    data.init_data()
    sc, fc = data.get_sc_fc('cg')
    # ==================================================
    for band in data.filters:
        if fc[band] is not None:
            fig = gen_causal_distribution_figure(fc[band], sc[band], None)
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
        else:
            print_log(f"Data for {band:s} band does not exist.", start)