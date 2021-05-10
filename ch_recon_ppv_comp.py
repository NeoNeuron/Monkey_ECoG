import numpy as np
import matplotlib.pyplot as plt

def plot_ppv_curve(ax, data:dict):
    separator = [-6, -5, -4, -3, -2, -1, 0]
    for roc_data, color, label in zip(data['roc_data'], data['colors'], data['labels']):
        ax.plot(separator, 100*roc_data[:, -1], '-o',
                markersize=2, markerfacecolor='None', color=color, label=label)
        # ax.plot(separator, 100*roc_data[:, i, -3], '-s',
        #         markerfacecolor='None', color=color, label='TPR'+label)

    ax.plot(separator, 100*(roc_data[:, 0]+roc_data[:,2])/roc_data[:,0:4].sum(1),
            '-o', markersize=2, markerfacecolor='None', color='gray', ls='--', label='p true')
    ax.grid(ls='--')
    ax.set_ylim(0,100)
    ax.set_ylabel('Percentage(%)')
    ax.set_xlabel(r'$\log_{10}$(SCs Thresholding)')
    return ax

if __name__ == '__main__':
    from utils.plot import plot_union
    path = 'tdmi_snr_analysis/'
    fnames = [
        path + 'tdmi_kmean/recon_gap_tdmi.npy', 
        path + 'tdmi_detrend_kmean/recon_gap_tdmi.npy',
        path + 'gc_6/recon_gap_gc.npy', 
        path + 'gc_detrend/recon_gap_gc.npy',
        path + 'cgc/recon_gap_gc.npy', 
        path + 'cc_linear_abs/recon_gap_cc.npy',
        path + 'tdcc_kmean/recon_gap_tdcc.npy',
        path + 'tdcc_detrend_kmean/recon_gap_tdcc.npy',
    ]
    all_data = [np.load(fname, allow_pickle=True) for fname in fnames]
    colors = ['red', 'orange', 'royalblue', 'm', 'y', 'k', 'cyan', 'springgreen',]
    labels = ['TDMI', 'TDMI_detrend', 'GC', 'GC_detrend', 'Cond GC', 'CC', 'TDCC', 'TDCC_detrend']
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    data_plt = {}
    for i, band in enumerate(filters):
        data_plt[band] = {
            'roc_data':[ data[:, i, :] for data in all_data ],
            'colors':colors,
            'labels':labels,
        }
    fig = plot_union(data_plt, plot_ppv_curve)
    plt.savefig(path + 'ch_bin_recon_ppv_comp_all_detrend.png')
    plt.close()