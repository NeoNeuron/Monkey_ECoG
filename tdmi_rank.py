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
    # plt.rcParams['xtick.labelsize'] = 16
    # plt.rcParams['ytick.labelsize'] = 16
    from utils.tdmi import MI_stats, compute_snr_matrix, compute_noise_matrix
    from utils.utils import print_log
    from utils.tdmi import find_gap_threshold
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import pickle
    arg_default = {
        'path': 'tdmi_snr_analysis/',
        'tdmi_mode': 'max',
        'is_interarea': False,
        'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
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
        'tdmi_mode', 
        default=arg_default['tdmi_mode'], nargs='?',
        type = str, choices=['max', 'sum'], 
        help = "TDMI mode."
    )
    parser.add_argument(
        'is_interarea', 
        default=arg_default['is_interarea'], nargs='?', 
        type=bool, 
        help = "inter-area flag."
    )
    parser.add_argument(
        '--filters', 
        default=arg_default['filters'], nargs='*', 
        type=str, 
        help = "list of filtering band."
    )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    # prepare weight_flatten
    weight = data_package['weight']
    off_diag_mask = ~np.eye(weight.shape[0], dtype=bool)
    with open(args.path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)
    
    fig, ax  = plt.subplots(2,4, figsize=(15,6), sharex=True)
    ax = ax.reshape((-1,))
    tdmi_data = np.load(args.path+'tdmi_data_long.npz', allow_pickle=True)

    gap_th_val= {}
    for idx, band in enumerate(args.filters):
        # generate snr mask
        snr_mat = compute_snr_matrix(tdmi_data[band])
        noise_matrix = compute_noise_matrix(tdmi_data[band])
        snr_mask = snr_mat >= snr_th[band]

        tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
        # apply snr mask
        tdmi_data_band[~snr_mask] = noise_matrix[~snr_mask]
        tdmi_data_flatten = tdmi_data_band[off_diag_mask]

        weight_flatten = weight[off_diag_mask]
        # setup interarea mask
        if args.is_interarea:
            interarea_mask = (weight_flatten != 1.5)
            weight_flatten = weight_flatten[interarea_mask]
            tdmi_data_flatten = tdmi_data_flatten[interarea_mask]

        gap_th_val[band] = find_gap_threshold(np.log10(tdmi_data_flatten))
        ax[idx].plot(np.log10(np.sort(tdmi_data_flatten)), '.', ms=0.1)
        ax[idx].set_xlabel('Ranked TDMI index')
        ax[idx].set_ylabel(r'$\log_{10}$(TDMI value)')
        ax[idx].set_title(band)
        ax[idx].axhline(gap_th_val[band], color='orange')
        print(gap_th_val[band])
        # axt=ax[idx].twinx()
        # axt.plot(np.log10(weight_flatten[np.argsort(tdmi_data_flatten)]), '.', color='orange', ms=0.1)
        # axt.set_ylabel(r'$\log_{10}$(weight)')
        print_log(f"Figure {band:s} generated.", start)

    ax[-1].plot(np.log10(np.sort(weight_flatten)), '.', color='orange', ms=0.1)
    ax[-1].set_xlabel('Ranked weight index')
    ax[-1].set_ylabel(r'$\log_{10}$(weight)')
    ax[-1].set_title('Weight')
    plt.tight_layout()

    with open(args.path+'gap_th.pkl', 'wb') as f:
        pickle.dump(gap_th_val, f)

    if args.is_interarea:
        fname = f'channel_tdmi_rank_interarea_{args.tdmi_mode:s}_manual-th.png'
    else:
        fname = f'channel_tdmi_rank_{args.tdmi_mode:s}_manual-th.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)