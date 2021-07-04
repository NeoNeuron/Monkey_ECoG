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
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    from fcpy.core import EcogTDMI
    from fcpy.plot import gen_sc_fc_figure
    from fcpy.utils import print_log
    from fcpy.binary_threshold import find_gap_threshold
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
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    dist_mat = data_package['d_matrix']
    sc, fc = data.get_sc_fc('ch')
    snr_mask = data.get_snr_mask(args.path)
    roi_mask = data.roi_mask.copy()
    # ==================================================
    
    d_mat = {}
    for band in data.filters:
        # setup interarea mask
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            weight_flatten = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]
        d_mat[band] = dist_mat[roi_mask]
    fig = gen_sc_fc_figure(fc, d_mat, tdmi_threshold=None,snr_mask=snr_mask)
    ax = fig.get_axes()
    [axi.set_xlabel(r'$\log_{10}$(dist)') for axi in ax]
    # [axi.set_ylabel(r'$\log_{10}$(Dist)') for axi in ax]
    [axi.set_ylabel(r'$\log_{10}g$(MI)') for axi in ax]
    # [ax[i].get_lines()[2].set_color((0,0,0,0)) for i in range(7)]
    handles, labels = ax[0].get_legend_handles_labels()
    ax[-1].legend(handles, labels)
    ax[-1].axis('off')

    if args.is_interarea:
        fname = f'ch_interarea_analysis_mi_snr_reg.png'
    else:
        fname = f'ch_analysis_mi_snr_reg.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)

