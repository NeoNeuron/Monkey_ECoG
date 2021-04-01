# linear fitting between weight(log) and tdmi value(log).
# pairs with sufficient SNR value are counted.

if __name__ == '__main__':
    from utils.binary_threshold import *
    from utils.core import EcogTDMI
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

    # Load SC and FC data
    # ==================================================
    data = EcogTDMI('data/')
    data.init_data(args.path)
    sc, fc = data.get_sc_fc('cg')

    data = EcogTDMI('data/')
    data.init_data_strict(args.path)
    _, fc_fit = data.get_sc_fc('cg')
    # ==================================================
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    fit_th = get_fit_threshold(sc, fc_fit, w_thresholds)
    gap_th = get_gap_threshold(fc)
    roc_th = get_roc_threshold(sc, fc, w_thresholds)

    suffix = '_tdmi_CG'
    with open(args.path + 'th_fit'+suffix+'.pkl', 'wb') as f:
        pickle.dump(fit_th, f)
    with open(args.path + 'th_roc'+suffix+'.pkl', 'wb') as f:
        pickle.dump(roc_th, f)
    with open(args.path + 'th_gap'+suffix+'.pkl', 'wb') as f:
        pickle.dump(gap_th, f)