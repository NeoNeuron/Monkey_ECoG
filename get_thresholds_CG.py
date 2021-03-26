# linear fitting between weight(log) and tdmi value(log).
# pairs with sufficient SNR value are counted.

if __name__ == '__main__':
    from utils.binary_threshold import *
    from utils.tdmi import MI_stats, compute_snr_matrix, compute_noise_matrix
    from utils.utils import CG
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    import pickle
    arg_default = {
        'path': 'tdmi_snr_analysis/',
    }
    parser = ArgumentParser(
        description = "Generate three types of thresholding criteria.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path', default=arg_default['path'], nargs='?',
        type = str, 
        help = "path of working directory."
    )
    args = parser.parse_args()

    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    # load and manipulate weight matrix
    weight = data_package['adj_mat']
    weight[weight == 0] = 1e-6
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    weight[np.eye(weight.shape[0], dtype=bool)] = 1.5
    weight[~cg_mask] = np.nan

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)

    weight_flatten = weight[cg_mask]
    weight_flatten = {band:weight_flatten for band in filter_pool}

    # load SNR thresholds
    with open(args.path+'snr_th.pkl', 'rb') as f:
        snr_th = pickle.load(f)

    # prepare TDMI data
    tdmi_flatten_fit = {}
    tdmi_flatten = {}
    for band in filter_pool:
        snr_matrix = compute_snr_matrix(tdmi_data[band])
        noise_matrix = compute_noise_matrix(tdmi_data[band])
        snr_mask = snr_matrix >= snr_th[band]
        tdmi_band = MI_stats(tdmi_data[band], 'max')

        tdmi_band[~snr_mask] = noise_matrix[~snr_mask]
        tdmi_flatten[band] = CG(tdmi_band, stride)
        tdmi_flatten[band] = tdmi_flatten[band][cg_mask]

        tdmi_band[~snr_mask] = np.nan
        tdmi_flatten_fit[band] = CG(tdmi_band, stride)
        tdmi_flatten_fit[band] = tdmi_flatten_fit[band][cg_mask]

    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    fit_th = get_fit_threshold(weight_flatten, tdmi_flatten_fit, w_thresholds)
    gap_th = get_gap_threshold(tdmi_flatten, 1000)
    roc_th = get_roc_threshold(weight_flatten, tdmi_flatten, w_thresholds)

    suffix = '_tdmi_CG'
    with open(args.path + 'th_fit'+suffix+'.pkl', 'wb') as f:
        pickle.dump(fit_th, f)
    with open(args.path + 'th_roc'+suffix+'.pkl', 'wb') as f:
        pickle.dump(roc_th, f)
    with open(args.path + 'th_gap'+suffix+'.pkl', 'wb') as f:
        pickle.dump(gap_th, f)