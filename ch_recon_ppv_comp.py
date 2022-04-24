import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from fcpy.plot_frame import plot_union
    from fcpy.plot import plot_ppv_curve
    path = 'tdmi_snr_analysis/'
    fnames = [
        path + 'tdmi_kmean_ty9/recon_gap_tdmi.npy', 
        # path + 'tdmi_detrend_kmean/recon_gap_tdmi.npy',
        path + 'gc_6_ty9/recon_gap_gc.npy', 
        # path + 'gc_detrend/recon_gap_gc.npy',
        path + 'cgc_ty9/recon_gap_gc.npy', 
        path + 'cc_ty9/recon_gap_cc.npy',
        path + 'tdcc_kmean_ty9/recon_gap_tdcc.npy',
        # path + 'tdcc_detrend_kmean/recon_gap_tdcc.npy',
    ]
    all_data = [np.load(fname, allow_pickle=True) for fname in fnames]
    colors = ['red', 'royalblue', 'm', 'y', 'springgreen',]
    labels = ['TDMI', 'GC', 'Cond GC', 'CC', 'TDCC',]
    filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw', 'sub_delta', 'above_delta']
    data_plt = {
        band: {
            'roc_data':[ data[:, i, :] for data in all_data ],
            'colors':colors,
            'labels':labels,
        }
        for i, band in enumerate(filters)
    }
    fig = plot_union(data_plt, plot_ppv_curve)
    plt.savefig(path + 'ch_bin_recon_ppv_comp_all_ty9.png')
    plt.close()