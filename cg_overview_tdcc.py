#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

from utils.plot import gen_sc_fc_figure


if __name__ == '__main__':
    import time
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from utils.core import EcogTDCC
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'tdmi_snr_analysis/',
                   }
    parser = ArgumentParser(prog='gc_analysis',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogTDCC()
    data.init_data()
    sc, fc = data.get_sc_fc('cg')
    # ==================================================

    fig = gen_sc_fc_figure(fc, sc, None, is_log=False)
    ax = fig.get_axes()

    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_ylabel('CC') for i in (0,2,4,6)]
    [ax[i].set_xlabel(r'$\log_{10}$(SC)') for i in (5,6)]
    handles, labels = ax[0].get_legend_handles_labels()
    labels = [item.replace('TDMI', 'CC') for item in labels]
    ax[-1].legend(handles, labels)
    ax[-1].axis('off')

    fname = f'cg_sc_fc_tdcc.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)