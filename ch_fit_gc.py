#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt 
    plt.rcParams['font.size']=15
    plt.rcParams['axes.labelsize'] = 15
    from fcpy.core import EcogGC
    from fcpy.plot import gen_mi_s_figure
    from fcpy.utils import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                    'order': 6,
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='gc_s',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('order', default=arg_default['order'], nargs='?',
                        type = int,
                        help = "order of regression model in GC."
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogGC()
    data.init_data()
    sc, fc = data.get_sc_fc('ch')
    # ==================================================
    if args.is_interarea:
        for band in data.filters:
            if fc[band] is not None:
                interarea_mask = (sc[band] != 1.5)
                sc[band] = sc[band][interarea_mask]
                fc[band] = fc[band][interarea_mask]

    fig = gen_mi_s_figure(fc, sc)

    # edit axis labels
    for ax in fig.get_axes():
        handles, labels = ax.get_legend_handles_labels()
        labels = [item.replace('TDMI', 'GC') for item in labels]
        ax.legend(handles, labels)
    [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(GC\right)$') for i in (0,4)]
    [fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
    plt.tight_layout()

    if args.is_interarea:
        fname = f'gc-s_interarea_{args.order:d}.png'
    else:
        fname = f'gc-s_{args.order:d}.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)