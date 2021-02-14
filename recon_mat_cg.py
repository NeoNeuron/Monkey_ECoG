#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot the reconstructed adjacent matrix of whole brain

if __name__ == '__main__':
    import numpy as np
    import matplotlib as mpl 
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from draw_causal_distribution_v2 import load_data
    from CG_causal_distribution import Extract_MI_CG

    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    adj_mat = data_package['adj_mat']
    multiplicity = data_package['multiplicity']
    stride = data_package['stride']

    filter_pool = ['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

    tdmi_mode = 'sum'  # or 'max'

    # create adj_weight_flatten by excluding 
    #   auto-tdmi in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    n_threshold = 6
    fig = plt.figure(constrained_layout=False, figsize=(14,12), dpi=100)
    gs = fig.add_gridspec(nrows=len(filter_pool),ncols=n_threshold+1, 
                          left=0.02, right=0.98, top=0.99, bottom=0.01, 
                          wspace=0.01)
    ax = np.array([fig.add_subplot(i) for i in gs])
    ax = ax.reshape((len(filter_pool), n_threshold+1))
    
    for idx, band in enumerate(filter_pool):
        # plot true adjacent matrix
        if idx == 0:
            pax = ax[idx, 0].imshow(np.log10(adj_mat+1e-6))
            ax[idx, 0].axis('scaled')
            divider = make_axes_locatable(ax[idx, 0])
            cax = divider.append_axes("left", size="5%", pad=0.05)
            plt.colorbar(pax, cax = cax)
            cax.yaxis.set_ticks_position('left')
            ax[idx, 0].yaxis.set_visible(False)
        else:
            ax[idx, 0].set_visible(False)

        # load tdmi data for target band
        tdmi_data = load_data(path, band)
        tdmi_data_cg = Extract_MI_CG(tdmi_data, tdmi_mode, stride, multiplicity)
        tdmi_data_cg = np.log10(tdmi_data_cg)

        # plot reconstructed connectome
        thresholds = np.linspace(tdmi_data_cg.min(), tdmi_data_cg.max(), n_threshold+2)
        thresholds = np.flip(thresholds[1:-1])
        lb = tdmi_data_cg.min()
        tdmi_data_cg[np.eye(tdmi_data_cg.shape[0], dtype=bool)] = tdmi_data_cg.min()

        for iidx, threshold in enumerate(thresholds):
            tdmi_data_cg_copy = tdmi_data_cg.copy()
            tdmi_data_cg_copy[tdmi_data_cg_copy<threshold] = lb

            pax1 = ax[idx, iidx+1].imshow(tdmi_data_cg_copy)
            ax[idx, iidx+1].axis('scaled')
            ax[idx, iidx+1].xaxis.set_visible(False)
            ax[idx, iidx+1].yaxis.set_visible(False)
            divider = make_axes_locatable(ax[idx, iidx+1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.set_visible(False)

        plt.colorbar(pax1, cax = cax)
        cax.set_visible(True)

    plt.savefig(path + f'recon_cg_{tdmi_mode:s}_all.png')
