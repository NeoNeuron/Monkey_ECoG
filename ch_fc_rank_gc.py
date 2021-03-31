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
    from utils.core import EcogGC
    from utils.utils import print_log
    from utils.binary_threshold import find_gap_threshold
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
        'is_interarea': False,
    }
    parser = ArgumentParser(
        prog='tdmi_snr_causal',
        description = "Plot ranked GC.",
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
    data = EcogGC('data/')
    data.init_data()
    sc, fc = data.get_sc_fc('ch')
    # ==================================================
    
    fig, ax  = plt.subplots(2,4, figsize=(15,6), sharex=True)
    ax = ax.reshape((-1,))

    for idx, band in enumerate(data.filters):
        # setup interarea mask
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            sc[band] = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]

        gap_th_val = find_gap_threshold(np.log10(fc[band]), 1000)
        ax[idx].plot(np.log10(np.sort(fc[band])), '.', ms=0.1)
        ax[idx].set_xlabel('Ranked GC index')
        ax[idx].set_ylabel(r'$\log_{10}$(GC value)')
        ax[idx].set_title(band)
        ax[idx].axhline(gap_th_val, color='orange')
        # axt=ax[idx].twinx()
        # axt.plot(np.log10(sc[np.argsort(fc[band])]), '.', color='orange', ms=0.1)
        # axt.set_ylabel(r'$\log_{10}$(weight)')
        print_log(f"Figure {band:s} generated.", start)

    ax[-1].plot(np.log10(np.sort(sc[band])), '.', color='orange', ms=0.1)
    ax[-1].set_xlabel('Ranked weight index')
    ax[-1].set_ylabel(r'$\log_{10}$(weight)')
    ax[-1].set_title('Weight')
    plt.tight_layout()

    if args.is_interarea:
        fname = f'ch_gc_rank_interarea_manual-th.png'
    else:
        fname = f'ch_gc_rank_manual-th.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)