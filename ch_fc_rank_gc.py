#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot ranked GC value, and calculate the gap threshold value.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 14
    # plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
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
    
    fig = plt.figure(figsize=(8,15), dpi=100)
    gs = fig.add_gridspec(nrows=4, ncols=2, 
                          left=0.10, right=0.90, top=0.96, bottom=0.05, 
                          wspace=0.36, hspace=0.30)
    ax = np.array([fig.add_subplot(i) for i in gs])
    axt = []

    for idx, band in enumerate(data.filters):
        # setup interarea mask
        if args.is_interarea:
            interarea_mask = (sc[band] != 1.5)
            sc[band] = sc[band][interarea_mask]
            fc[band] = fc[band][interarea_mask]

        gap_th_val, gap_th_label = find_gap_threshold(np.log10(fc[band]))
        ax[idx].plot(np.log10(np.sort(fc[band])), np.arange(fc[band].shape[0]), '.', ms=0.1)
        ax[idx].yaxis.get_major_formatter().set_powerlimits((0,1))
        ax[idx].set_title(band)
        ax[idx].axvline(gap_th_val, color='r', label=gap_th_label)
        axt.append(ax[idx].twinx())
        axt[idx].hist(np.log10(fc[band][sc[band]>0]),color='orange', alpha=.5, bins=100, label='SC Present')
        axt[idx].hist(np.log10(fc[band][sc[band]==0]), color='navy', alpha=.5, bins=100, label='SC Absent')
        axt[idx].yaxis.get_major_formatter().set_powerlimits((0,1))
        ax[idx].legend(fontsize=10, loc=5)
        ax[idx].text(
            0.05, 0.95, 
            f'PPV:{np.sum(fc[band][sc[band]>0]>10**gap_th_val)*100./np.sum(fc[band]>10**gap_th_val):4.1f} %',
            fontsize=14, transform=ax[idx].transAxes, 
            verticalalignment='top', horizontalalignment='left'
        )

    [ax[i].set_ylabel('Ranked GC index') for i in (0,2,4,6)]
    [ax[i].set_xlabel(r'$\log_{10}$(GC value)') for i in (5,6)]
    [axt[i].set_ylabel('Counts') for i in (1,3,5)]
    handles, labels = axt[0].get_legend_handles_labels()
    ax[-1].legend(handles, labels, fontsize=16, loc=2)
    ax[-1].axis('off')

    if args.is_interarea:
        fname = f'ch_gc_rank_interarea.png'
    else:
        fname = f'ch_gc_rank.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)