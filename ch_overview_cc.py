#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

if __name__ == '__main__':
    import time
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from utils.core import EcogCC
    from utils.plot import gen_causal_distribution_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
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
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogCC()
    data.init_data()
    sc, fc = data.get_sc_fc('ch')
    # ==================================================
    for band in data.filters:
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            sc[band] = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]

        fig = gen_causal_distribution_figure(fc[band], sc[band], None, is_log=False)

        fig.get_axes()[4].set_ylabel('CC')
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        labels = [item.replace('TDMI', 'CC') for item in labels]
        fig.get_axes()[4].legend(handles, labels)
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        if args.is_interarea:
            fname = f'ch_{band:s}_cc_interarea.png'
        else:
            fname = f'ch_{band:s}_cc.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)