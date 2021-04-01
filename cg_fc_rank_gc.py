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
    # plt.rcParams['xtick.labelsize'] = 16
    # plt.rcParams['ytick.labelsize'] = 16
    from utils.core import EcogGC
    from utils.utils import print_log
    from utils.binary_threshold import find_gap_threshold
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {
        'path': 'tdmi_snr_analysis/',
    }
    parser = ArgumentParser(
        prog='tdmi_rank_cg',
        description = "Plot ranked GC.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path', default=arg_default['path'], nargs='?',
        type = str, 
        help = "path of working directory."
    )
    args = parser.parse_args()

    start = time.time()
    # Load SC and FC data
    # ==================================================
    data = EcogGC('data/')
    data.init_data()
    sc, fc = data.get_sc_fc('cg')
    # ==================================================
    
    fig, ax  = plt.subplots(2,4, figsize=(15,6), sharex=True)
    ax = ax.reshape((-1,))

    for idx, band in enumerate(data.filters):
        gap_th_val, gap_th_label = find_gap_threshold(np.log10(fc[band]))
        ax[idx].plot(np.log10(np.sort(fc[band])), '.', ms=0.1)
        ax[idx].set_xlabel('Ranked GC index')
        ax[idx].set_ylabel(r'$\log_{10}$(GC value)')
        ax[idx].set_title(band)
        ax[idx].axhline(gap_th_val, color='orange', label=gap_th_label)
        # from scipy.optimize import curve_fit
        # from utils.utils import Gaussian, Double_Gaussian
        # axt=ax[idx].twiny()
        # (counts, edges) = np.histogram(np.log10(fc[band]), bins=100)
        # popt, _ = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
        # if popt[2] > popt[3]:
        #     axt.plot(Gaussian(edges[1:], popt[0],popt[2],popt[4]), edges[1:], 'ro', markersize = 3, label=r'$1^{st}$ Gaussian fit')
        #     axt.plot(Gaussian(edges[1:], popt[1],popt[3],popt[5]), edges[1:], 'bo', markersize = 3, label=r'$2^{nd}$ Gaussian fit')
        # else:
        #     axt.plot(Gaussian(edges[1:], popt[0],popt[2],popt[4]), edges[1:], 'bo', markersize = 3, label=r'$2^{nd}$ Gaussian fit')
        #     axt.plot(Gaussian(edges[1:], popt[1],popt[3],popt[5]), edges[1:], 'ro', markersize = 3, label=r'$1^{nd}$ Gaussian fit')
        # axt.set_xlabel('Counts')
        ax[idx].legend()
        print_log(f"Figure {band:s} generated.", start)

    ax[-1].plot(np.log10(np.sort(sc[band])), '.', color='orange', ms=0.1)
    ax[-1].set_xlabel('Ranked weight index')
    ax[-1].set_ylabel(r'$\log_{10}$(weight)')
    ax[-1].set_title('Weight')
    plt.tight_layout()

    fname = f'cg_gc_rank.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)
    plt.close(fig)