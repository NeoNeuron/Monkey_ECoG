# linear fitting between weight(log) and tdmi value(log).
# pairs with sufficient SNR value are counted.

if __name__ == '__main__':
    from utils.binary_threshold import *
    import pickle
    import numpy as np
    from utils.core import EcogTDCC
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    import pickle
    arg_default = {'path': 'tdmi_snr_analysis/'}
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
    # Load SC and FC data
    # ==================================================
    data = EcogTDCC('data/')
    data.init_data(args.path, 'snr_th_gauss_tdcc.pkl')
    sc, fc = data.get_sc_fc('ch')
    # ==================================================

    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    fit_th = get_fit_threshold(sc, fc, w_thresholds, is_log=False)
    gap_th = get_gap_threshold(fc, is_log=False)
    roc_th = get_roc_threshold(sc, fc, w_thresholds, is_log=False)

    suffix = '_tdcc'
    with open(args.path + 'th_fit'+suffix+'.pkl', 'wb') as f:
        pickle.dump(fit_th, f)
    with open(args.path + 'th_roc'+suffix+'.pkl', 'wb') as f:
        pickle.dump(roc_th, f)
    with open(args.path + 'th_gap'+suffix+'.pkl', 'wb') as f:
        pickle.dump(gap_th, f)