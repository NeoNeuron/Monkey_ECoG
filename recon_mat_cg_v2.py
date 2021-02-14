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

    adj_mat_log = np.log10(adj_mat+1e-6)  # add small epsilon to avoid nan in log10

    n_threshold = 5
    fig = plt.figure(constrained_layout=False, figsize=(10,12), dpi=100)
    gs = fig.add_gridspec(nrows=len(filter_pool)+1, ncols=n_threshold, 
                          left=0.02, right=0.98, top=0.99, bottom=0.01, 
                          wspace=0.01)
    ax = np.array([fig.add_subplot(i) for i in gs])
    ax = ax.reshape((len(filter_pool)+1, n_threshold))

    # plot binary connectome for original matrix
    thresholds=[-5,-4,-3,-2,-1]
    for idx, threshold in enumerate(thresholds):
        adj_mat_binary = (adj_mat_log>threshold)
        pax = ax[0, idx].imshow(adj_mat_binary.astype(int), cmap='gist_gray')
        ax[0, idx].axis('scaled')
        ax[0, idx].xaxis.set_visible(False)
        ax[0, idx].yaxis.set_visible(False)
    
    optimal_threshold = np.load(path+f'opt_threshold_{tdmi_mode:s}.npz', allow_pickle=True)
    for idx, band in enumerate(filter_pool):
        # load shuffled tdmi data for target band
        tdmi_data = load_data(path, band)
        thresholds = optimal_threshold[band][1:-1]
        print(thresholds)
        
        tdmi_data_cg = Extract_MI_CG(tdmi_data, tdmi_mode, stride, multiplicity)
        tdmi_data_cg = np.log10(tdmi_data_cg)

        # plot reconstructed connectome
        lb = tdmi_data_cg.min()
        tdmi_data_cg[np.eye(tdmi_data_cg.shape[0], dtype=bool)] = tdmi_data_cg.min()

        for iidx, threshold in enumerate(thresholds):
            tdmi_data_cg_binary = (tdmi_data_cg>threshold)

            pax1 = ax[idx+1, iidx].imshow(tdmi_data_cg_binary, cmap='gist_gray')
            ax[idx+1, iidx].axis('scaled')
            ax[idx+1, iidx].xaxis.set_visible(False)
            ax[idx+1, iidx].yaxis.set_visible(False)
     
    plt.savefig(path + f'recon_cg_{tdmi_mode:s}_all_binary.eps')