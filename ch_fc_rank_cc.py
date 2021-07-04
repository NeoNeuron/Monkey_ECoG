#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot ranked Correlation Coefficient value, and calculate the gap threshold value.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 14
    # plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    from fcpy.core import EcogCC
    from fcpy.utils import print_log
    from fcpy.plot import gen_fc_rank_figure_single
    from fcpy.plot_frame import *
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
        'is_interarea': False,
    }
    parser = ArgumentParser(
        prog='tdmi_snr_causal',
        description = "Plot ranked TDMI.",
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
    data = EcogCC('data/')
    data.init_data()
    sc, fc = data.get_sc_fc('ch')
    # ==================================================
    data_plt = {}
    for band in data.filters:
        data_plt[band] = {
            'fc':fc[band],
            'sc':sc[band],
            'band':band,
            'is_log':False,
        }

    fig = fig_frame33(data_plt, gen_fc_rank_figure_single)
    ax = fig.get_axes()
    [axi.set_ylabel('Ranked CC index') for axi in ax if axi.get_ylabel()]
    [axi.set_xlabel('CC value') for axi in ax if axi.get_xlabel()]

    if args.is_interarea:
        fname = f'ch_cc_rank_interarea.png'
    else:
        fname = f'ch_cc_rank.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)