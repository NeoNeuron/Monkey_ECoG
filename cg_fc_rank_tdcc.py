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
    from utils.core import EcogTDCC
    from utils.utils import print_log
    from utils.plot import gen_fc_rank_figure
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
    data = EcogTDCC('data/')
    data.init_data(args.path, 'snr_th_gauss_tdcc.pkl')
    sc, fc = data.get_sc_fc('cg')
    # ==================================================
    
    fig = gen_fc_rank_figure(sc, fc, is_log=False)

    ax = fig.get_axes()
    [ax[i].set_ylabel('Ranked CC index') for i in (0,2,4,6)]
    [ax[i].set_xlabel('CC value') for i in (5,6)]

    if args.is_interarea:
        fname = f'cg_tdcc_rank_interarea.png'
    else:
        fname = f'cg_tdcc_rank.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)