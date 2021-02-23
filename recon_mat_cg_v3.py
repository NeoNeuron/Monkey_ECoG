#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot the reconstructed adjacent matrix of whole brain

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.tdmi import Extract_MI_CG
    from cluster import get_cluster_id, get_sorted_mat

    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    adj_mat = data_package['adj_mat']
    stride = data_package['stride']

    adj_mat_sort = adj_mat.copy()
    adj_mat_sort[adj_mat_sort<5e-3] = 1e-6
    sort_id = get_cluster_id(adj_mat_sort)

    filter_pool = ['raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

    tdmi_mode = 'max'  # or 'max'

    adj_mat_log = np.log10(adj_mat+1e-6)  # add small epsilon to avoid nan in log10

    n_threshold = 7
    fig = plt.figure(constrained_layout=False, figsize=(16,4), dpi=100)
    gs = fig.add_gridspec(nrows=2, ncols=n_threshold, 
                          left=0.02, right=0.98, top=0.99, bottom=0.01, 
                          wspace=0.01)
    ax = np.array([fig.add_subplot(i) for i in gs])
    ax = ax.reshape((2, n_threshold))

    # plot binary connectome for original matrix
    thresholds=[-6,-5,-4,-3,-2,-1]
    for idx, threshold in enumerate(thresholds):
        adj_mat_binary = (adj_mat_log>threshold)
        adj_mat_binary = get_sorted_mat(adj_mat_binary, sort_id)
        pax = ax[0, idx].imshow(adj_mat_binary, cmap='gist_gray')
        ax[0, idx].axis('scaled')
        ax[0, idx].xaxis.set_visible(False)
        ax[0, idx].yaxis.set_visible(False)
    ax[0,-1].set_visible(False)
    optimal_threshold = np.load(path+f'opt_threshold_{tdmi_mode:s}.npz', allow_pickle=True)
    tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
    for idx, band in enumerate(filter_pool):
        # load shuffled tdmi data for target band
        thresholds = optimal_threshold[band][1:-1]
        threshold = thresholds.mean()
        
        tdmi_data_cg = Extract_MI_CG(tdmi_data[band], tdmi_mode, stride)
        tdmi_data_cg = np.log10(tdmi_data_cg)

        # plot reconstructed connectome
        lb = tdmi_data_cg.min()
        tdmi_data_cg[np.eye(tdmi_data_cg.shape[0], dtype=bool)] = tdmi_data_cg.min()

        tdmi_data_cg_binary = (tdmi_data_cg>threshold)
        tdmi_data_cg_binary = get_sorted_mat(tdmi_data_cg_binary, sort_id)

        pax1 = ax[1, idx].imshow(tdmi_data_cg_binary, cmap='gist_gray')
        ax[1,idx].set_title(band)
        ax[1, idx].axis('scaled')
        ax[1, idx].xaxis.set_visible(False)
        ax[1, idx].yaxis.set_visible(False)
     
    plt.savefig(path + f'recon_cg_{tdmi_mode:s}_all_binary_short.eps')