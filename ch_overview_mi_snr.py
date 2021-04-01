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
    from utils.core import EcogTDMI
    from utils.plot import gen_causal_distribution_figure
    from utils.utils import print_log
    from utils.binary_threshold import find_gap_threshold
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
        'is_interarea': False,
    }
    parser = ArgumentParser(
        prog='tdmi_snr_causal',
        description = "Generate figure for analysis of causality.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path', default=arg_default['path'], nargs='?',
        type = str, 
        help = "path of working directory."
    )
    parser.add_argument(
        'is_interarea', 
        default=arg_default['is_interarea'], nargs='?', 
        type=bool, 
        help = "inter-area flag."
    )
   
    args = parser.parse_args()

    start = time.time()

    # Load SC and FC data
    # ==================================================
    data = EcogTDMI('data/')
    data.init_data(args.path)
    sc, fc = data.get_sc_fc('ch')
    snr_mask = data.get_snr_mask(args.path)
    # ==================================================
    
    for band in data.filters:
        # setup interarea mask
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            weight_flatten = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]
        fig = gen_causal_distribution_figure(
            fc[band], sc[band], tdmi_threshold=None,
            snr_mask=snr_mask[band]
        )
        gap_th, _ = find_gap_threshold(np.log10(fc[band]))
        fig.get_axes()[0].axvline(gap_th, ls='-', color='royalblue', label='gap')
        fig.get_axes()[4].axhline(gap_th, ls='-', color='royalblue', label='gap')
        fig.get_axes()[0].legend(fontsize=12)
        fig.get_axes()[4].legend(fontsize=12)


        fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        if args.is_interarea:
            fname = f'ch_{band:s}_interarea_analysis_snr.png'
        else:
            fname = f'ch_{band:s}_analysis_snr.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)
