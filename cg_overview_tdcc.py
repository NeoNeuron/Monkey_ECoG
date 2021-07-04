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
    from fcpy.core import EcogTDCC
    from fcpy.plot import gen_sc_fc_figure_single
    from fcpy.plot_frame import *
    from fcpy.utils import print_log
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
    data.init_data(args.path, 'snr_th_gauss_tdcc.pkl')
    sc, fc = data.get_sc_fc('cg')
    # ==================================================
    data_plt = {}
    for band in data.filters:
        data_plt[band] = {
            'fc':fc[band],
            'sc':sc[band],
            'band':band,
            'is_log':False,
        }
    fig = fig_frame52(data_plt, gen_sc_fc_figure_single)
    ax = fig.get_axes()

    [axi.set_ylabel(axi.get_ylabel().replace('TDMI', 'CC'))
        for axi in ax if axi.get_ylabel()]
    handles, labels = ax[0].get_legend_handles_labels()
    labels = [item.replace('TDMI', 'CC') for item in labels]
    ax[-1].legend(handles, labels)
    ax[-1].axis('off')

    fname = f'cg_sc_fc_tdcc.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)