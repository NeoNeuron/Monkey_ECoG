import numpy as np
import matplotlib.pyplot as plt

def plot_ppv_curves(fnames:list, labels:list, colors:list, figname:str):
    fig, ax = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    separator = [-6, -5, -4, -3, -2, -1, 0]
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

    for fname, color, label in zip(fnames, colors, labels):
        roc_data = np.load(fname, allow_pickle=True)
        for i, index in enumerate(indices):
            ax[index].plot(separator, 100*roc_data[:, i, -1], '-o',
                        markersize=2, markerfacecolor='None', color=color, label=label)
            # ax[index].plot(separator, 100*roc_data[:, i, -3], '-s',
            #              markerfacecolor='None', color=color, label='TPR'+label)

    for i, index in enumerate(indices):
        ax[index].plot(separator, 100*(roc_data[:,i, 0]+roc_data[:,i,2])/roc_data[:,i,0:4].sum(1),
                     '-o', markersize=2, markerfacecolor='None', color='k', label='p true')
        ax[index].grid(ls='--')
        ax[index].set_title(filters[i])
        ax[index].set_ylim(0,85)

    # plot legend in the empty subplot
    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[-1, -1].legend(handles, labels, loc=1, fontsize=16)
    ax[-1, -1].axis('off')

    [ax[i, 0].set_ylabel('Percentage(%)',fontsize=16) for i in (0, 1)]
    [ax[-1, i].set_xlabel(r'$\log_{10}$(Weight thresholding)',fontsize=12) for i in [0, 1, 2]]

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()

if __name__ == '__main__':
    path = 'tdmi_snr_analysis/'
    fnames = [
        path + 'snr_th_kmean/recon_gap_tdmi.npy', 
        path + 'snr_th_gauss/recon_gap_tdmi.npy',
        path + 'gc_6/recon_gap_gc.npy', 
        path + 'cc/recon_gap_gc.npy',
    ]
    colors = ['r', 'orange', 'royalblue', 'springgreen']
    labels = ['TDMI(KMeans)','TDMI(Gauss)', 'GC', 'CC' ]
    plot_ppv_curves(fnames, labels, colors, path + 'ch_bin_recon_ppv_comp.png')