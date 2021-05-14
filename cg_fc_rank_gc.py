#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot ranked GC value, and calculate the gap threshold value.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 14
    # plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    from utils.core import EcogGC
    from utils.utils import print_log
    from utils.plot import gen_fc_rank_figure_single
    from utils.plot_frame import *
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
    }
    parser = ArgumentParser(
        prog='tdmi_rank_cg',
        description = "Plot ranked GC.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path', default=arg_default['path'], nargs='?',
        type = str, 
        help = "path of working directory."
    )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogGC('data/')
    data.init_data()
    sc, fc = data.get_sc_fc('cg')
    # ==================================================
    data_plt = {}
    for band in data.filters:
        data_plt[band] = {
            'fc':fc[band],
            'sc':sc[band],
            'band':band,
        }

    fig = fig_frame33(data_plt, gen_fc_rank_figure_single)

    ax = fig.get_axes()
    [axi.set_ylabel('Ranked GC index') for axi in ax if axi.get_ylabel()]
    [axi.set_xlabel(r'$\log_{10}$(GC value)') for axi in ax if axi.get_xlabel()]

    fname = f'cg_gc_rank.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)