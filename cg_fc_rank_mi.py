#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot ranked TDMI value, and calculate the gap threshold value.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 14
    # plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    from utils.core import EcogTDMI
    from utils.utils import print_log
    from utils.plot import gen_fc_rank_figure
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
    }
    parser = ArgumentParser(
        prog='tdmi_rank_cg',
        description = "Plot ranked TDMI.",
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
    data = EcogTDMI('data/')
    data.init_data(args.path)
    sc, fc = data.get_sc_fc('cg')
    # ==================================================
    
    fig = gen_fc_rank_figure(sc, fc)

    ax = fig.get_axes()
    [ax[i].set_ylabel('Ranked TDMI index') for i in (0,2,4,6)]
    [ax[i].set_xlabel(r'$\log_{10}$(TDMI value)') for i in (5,6)]

    fname = f'cg_mi_rank.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)