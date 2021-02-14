#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np

def CG(gc_data:np.ndarray, stride:np.ndarray, multiplicity:np.ndarray=None)->np.ndarray:
    """Compute the coarse-grained average of 
        each cortical region for gc_data.

    Args:
        gc_data (np.ndarray): channel-wise gc_data.
        stride (np.ndarray): stride of channels. 
            Equal to the `cumsum` of multiplicity.
        multiplicity (np.ndarray, optional): number of channel
            in each cortical region. Defaults to None.

    Returns:
        np.ndarray: coarse-grained average of gc_data
    """
    if multiplicity is None:
        multiplicity = np.diff(stride).astype(int)
    n_region = stride.shape[0]-1
    gc_data_cg = np.zeros((n_region, n_region))
    for i in range(n_region):
        for j in range(n_region):
            data_buffer = gc_data[stride[i]:stride[i+1],stride[j]:stride[j+1]]
            if i != j:
                gc_data_cg[i,j]=data_buffer.mean()
            else:
                if multiplicity[i] > 1:
                    gc_data_cg[i,j]=np.mean(data_buffer[~np.eye(multiplicity[i], dtype=bool)])
                else:
                    gc_data_cg[i,j]=data_buffer.mean() # won't be used in ROC.
    return gc_data_cg

if __name__ == '__main__':
    import matplotlib as mpl 
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import ROC_curve, AUC, Youden_Index
    from gc_analysis import load_data

    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    multiplicity = data_package['multiplicity']
    stride = data_package['stride']
    n_region = multiplicity.shape[0]

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

    order = 10
    # create adj_weight_flatten by excluding 
    #   auto-gc in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    for band in filter_pool:
        # load shuffled gc data for target band
        gc_data, gc_data_shuffle = load_data(path, band, order, shuffle=True)
        gc_data_cg = CG(gc_data, stride, multiplicity)

        gc_data_flatten = gc_data_cg[cg_mask]
        gc_data_flatten[gc_data_flatten<=0] = 1e-5
        log_gc_data = np.log10(gc_data_flatten)
        log_gc_range = [log_gc_data.min(), log_gc_data.max()]

        # calculate histogram
        (counts, edges) = np.histogram(log_gc_data, bins=100, density=True)

        fig, ax = plt.subplots(2,4,figsize=(20,10))

        SI_value = gc_data_shuffle[~np.eye(stride[-1], dtype=bool)]
        SI_value[SI_value<=0] = 0
        SI_value = SI_value.mean()
        ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
        ax[0,0].axvline(np.log10(SI_value), color='cyan', label='SI')
        # UNCOMMENT to create double Gaussian fitting of gc PDF
        # # import fitting function
        # from scipy.optimize import curve_fit
        # from draw_causal_distribution_v2 import Gaussian, Double_Gaussian
        # try:
        #     popt, _ = curve_fit(Double_Gaussian, edges[1:], counts, p0=[0,0,0,0,1,1])
        #     ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[0],popt[2],popt[4]), 'ro', markersize = 4, label=r'$1^{st}$ Gaussian fit')
        #     ax[0,0].plot(edges[1:], Gaussian(edges[1:], popt[1],popt[3],popt[5]), 'bo', markersize = 4, label=r'$2^{nd}$ Gaussian fit')
        # except:
        #     print(f'WARNING: Failed fitting the {band:s} band case.')
        #     pass
        ax[0,0].set_xlabel('$log_{10}(Value)$')
        ax[0,0].set_ylabel('Density')
        ax[0,0].legend(fontsize=13, loc=2)

        weight_set = np.unique(adj_weight_flatten)
        log_gc_data_mean = np.array([np.mean(log_gc_data[adj_weight_flatten==key]) for key in weight_set])
        weight_set[weight_set==0]=1e-6
        pval = np.polyfit(np.log10(weight_set), log_gc_data_mean, deg=1)
        ax[1,0].plot(np.log10(adj_weight_flatten+1e-8), log_gc_data, 'k.', label='gc samples')
        # ax[1,0].plot(np.log10(weight_set), log_gc_data_mean, 'm.', label='gc mean')
        ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
        ax[1,0].set_ylabel(r'$log_{10}\left(gc\right)$')
        ax[1,0].set_xlabel(r'$log_{10}$(Connectivity Strength)')
        ax[1,0].set_title(f'Fitting Slop = {pval[0]:5.3f}')
        ax[1,0].legend(fontsize=15)

        # Draw ROC curves
        threshold_options = [1e-1, 5e-3, 1e-4]
        opt_threshold = np.zeros(len(threshold_options))
        for idx, threshold in enumerate(threshold_options):
            answer = adj_weight_flatten.copy()
            ax[0,idx+1].semilogy(np.sort(answer))
            ax[0,idx+1].set_xlabel('Ranked connectivity strength')
            ax[0,idx+1].set_ylabel('Connectivity Strength')
            ax[0,idx+1].axhline(threshold, color='r', ls = '--')
            ax[0,idx+1].set_title(f'Threshold = {threshold:3.2e}')

            # Plot ROC curve
            answer = (answer>threshold).astype(bool)
            thresholds = np.linspace(log_gc_range[0],log_gc_range[1],100)
            fpr, tpr = ROC_curve(answer, log_gc_data, thresholds)
            Youden_index = Youden_Index(fpr, tpr)
            opt_threshold[idx] = thresholds[Youden_index]

            ax[1,idx+1].plot(fpr, tpr, 'navy')
            ax[1,idx+1].set_xlabel('False positive rate')
            ax[1,idx+1].set_ylabel('True positive rate')
            ax[1,idx+1].plot(range(2),range(2), '--', color='orange')
            ax[1,idx+1].set_xlim(0,1)
            ax[1,idx+1].set_ylim(0,1)
            ax[1,idx+1].set_title(f'AUC = {AUC(fpr, tpr):5.3f}')
        ax[0,0].axvline(opt_threshold.mean(), color='orange', label='opt_threshold')
        ax[0,0].legend(fontsize=13, loc=2)

        plt.tight_layout()
        plt.savefig(path + f'cg_{band:s}_gc_order_{order:d}.png')