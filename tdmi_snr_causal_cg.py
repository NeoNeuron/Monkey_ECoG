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
    from utils.binary_threshold import find_gap_threshold
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
    }
    parser = ArgumentParser(
        prog='CG_causal_distribution',
        description = "Generate figure for coarse-grain analysis of causality.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path', default=arg_default['path'], 
        nargs='?', type = str, 
        help = "path of working directory."
    )
    args = parser.parse_args()

    start = time.time()
    
    # Load SC and FC data
    # ==================================================
    data = EcogTDMI('data/')
    data.init_data(args.path)
    sc, fc = data.get_sc_fc('cg')
    # ==================================================

    for band in data.filters:
        fig = gen_causal_distribution_figure(fc[band], sc[band], None)

        gap_th = find_gap_threshold(np.log10(fc[band]), 500)
        fig.get_axes()[0].axvline(gap_th, ls='-', color='royalblue', label='gap')
        fig.get_axes()[4].axhline(gap_th, ls='-', color='royalblue', label='gap')
        fig.get_axes()[0].legend(fontsize=14)
        fig.get_axes()[4].legend(fontsize=14)


        fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        fig.get_axes()[4].get_lines()[1].set_color((0,0,0,0))
        fig.get_axes()[4].get_lines()[2].set_color((0,0,0,0))
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        del handles[1]
        del handles[1]
        del labels[1]
        del labels[1]
        fig.get_axes()[4].legend(handles, labels ,fontsize=14)
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        fname = f'cg_{band:s}_analysis_manual-th.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)