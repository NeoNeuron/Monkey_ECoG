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
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    from utils.tdmi import MI_stats 
    from utils.tdmi import compute_snr_matrix, compute_noise_matrix
    from utils.plot import gen_causal_distribution_figure
    from utils.utils import CG, print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import pickle
    arg_default = {
        'path': 'tdmi_snr_analysis/',
        'tdmi_mode': 'max',
        'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
    }
    parser = ArgumentParser(
        prog='CG_causal_distribution',
        description = "Generate figure for coarse-grain analysis of causality.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path', default=arg_default['path'], 
        nargs='?', type = str, 
        help = "path of working directory."
    )
    parser.add_argument(
        'tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
        type = str, choices=['max', 'sum'], 
        help = "TDMI mode."
    )
    parser.add_argument(
        '--filters', default=arg_default['filters'], 
        nargs='*', type=str, 
        help = "list of filtering band."
    )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    n_region = multiplicity.shape[0]

    # manually set snr threshold
    with open(args.path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)

    # create adj_weight_flatten by excluding 
    #   auto-tdmi in region with single channel
    adj_weight = data_package['adj_mat'] + \
        np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[~cg_mask]

    # load shuffled tdmi data for target band
    tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
    # tdmi_data_shuffle = np.load('data/tdmi_data_shuffle.npz', allow_pickle=True)

    for band in args.filters:
        # generate SNR mask
        snr_mat = compute_snr_matrix(tdmi_data[band])
        noise_matrix = compute_noise_matrix(tdmi_data[band])
        snr_mask = snr_mat >= snr_th[band]
        # compute TDMI statistics
        tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
        tdmi_data_band[~snr_mask] = noise_matrix[~snr_mask]
        # compute coarse-grain average
        tdmi_data_cg = CG(tdmi_data_band, stride)
        # apply cg mask
        tdmi_data_flatten = tdmi_data_cg[~cg_mask]

        SI_value = 0  # disabled
        # SI_value = tdmi_data_shuffle[band][~cg_mask].mean()
        # if args.tdmi_mode == 'sum':
        #     SI_value *= 10

        fig = gen_causal_distribution_figure(
            tdmi_data_flatten, 
            adj_weight_flatten,
            SI_value,
        )

        from utils.binary_threshold import find_gap_threshold
        gap_th = find_gap_threshold(np.log10(tdmi_data_flatten), 500)
        fig.get_axes()[0].axvline(gap_th, ls='-', color='royalblue', label='gap')
        fig.get_axes()[4].axhline(gap_th, ls='-', color='royalblue', label='gap')
        fig.get_axes()[0].legend(fontsize=14)
        fig.get_axes()[4].legend(fontsize=14)


        if args.tdmi_mode == 'sum':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
        elif args.tdmi_mode == 'max':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        fig.get_axes()[4].get_lines()[1].set_color((0,0,0,0))
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        del handles[1]
        del labels[1]
        fig.get_axes()[4].legend(handles, labels ,fontsize=14)
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        fname = f'cg_{band:s}_analysis_{args.tdmi_mode:s}_manual-th.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)