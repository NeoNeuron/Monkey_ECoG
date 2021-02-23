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
    from utils.tdmi import MI_stats, compute_snr_matrix, get_sparsity_threshold
    from utils.utils import print_log
    from utils.plot import gen_mi_s_figure
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'tdmi_snr_analysis/',
                    'tdmi_mode': 'max',
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='tdmi_snr_mi_s',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
                        type = str, choices=['max', 'sum'], 
                        help = "TDMI mode."
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']
    tdmi_data = np.load(args.path + 'tdmi_data.npz', allow_pickle=True)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data_flatten = {}
    weight_flatten = {}
    snr_th = {
        'raw'        :5.0,
        'delta'      :4.3,
        'theta'      :4.5,
        'alpha'      :4,
        'beta'       :5.,
        'gamma'      :11,
        'high_gamma' :11,
    }
    for band in filter_pool:
        if band in tdmi_data.keys():
            tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
            # generate snr mask
            snr_mat = compute_snr_matrix(tdmi_data[band])
            # th_val = get_sparsity_threshold(snr_mat, p = 0.6)
            # snr_mask = snr_mat >= th_val
            snr_mask = snr_mat >= snr_th[band]

            tdmi_data_flatten[band] = tdmi_data_band[(~np.eye(stride[-1], dtype=bool))*snr_mask]
            # setup interarea mask
            weight_flatten[band] = weight[(~np.eye(stride[-1], dtype=bool))*snr_mask]
            if args.is_interarea:
                interarea_mask = (weight_flatten[band] != 1.5)
                weight_flatten[band] = weight_flatten[band][interarea_mask]
                tdmi_data_flatten[band] = tdmi_data_flatten[band][interarea_mask]
        else:
            tdmi_data_flatten[band] = None
            weight_flatten[band] = None

    fig = gen_mi_s_figure(tdmi_data_flatten, weight_flatten)

    # edit axis labels
    if args.tdmi_mode == 'sum':
        [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$') for i in (0,4)]
    elif args.tdmi_mode == 'max':
        [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$') for i in (0,4)]
    [fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
    plt.tight_layout()

    if args.is_interarea:
        fname = f'mi-s_interarea_{args.tdmi_mode:s}_manual-th.png'
    else:
        fname = f'mi-s_{args.tdmi_mode:s}_manual-th.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)