#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    from utils.core import EcogTDMI
    from utils.plot import gen_causal_distribution_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'is_interarea': False,
                   }
    parser = ArgumentParser(prog='draw_causal_distribution',
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

    # load data for target band
    for band in data.filters:
        # setup interarea mask
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            sc[band] = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]

        fig = gen_causal_distribution_figure(fc[band], sc[band], None)

        fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        if args.is_interarea:
            fname = f'ch_{band:s}_interarea_analysis.png'
        else:
            fname = f'ch_{band:s}_analysis.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)