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
    from utils.plot import gen_causal_distribution_figure, gen_sc_fc_figure
    from utils.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'tdmi_snr_analysis/',
                   'is_interarea': False,
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
    data = EcogTDCC()
    data.init_data(args.path, 'snr_th_gauss_tdcc.pkl')
    sc, fc = data.get_sc_fc('ch')
    # ==================================================
    for band in data.filters:
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            sc[band] = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]

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

    if args.is_interarea:
        fname = f'ch_sc_fc_tdcc_interarea.png'
    else:
        fname = f'ch_sc_fc_tdcc.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)