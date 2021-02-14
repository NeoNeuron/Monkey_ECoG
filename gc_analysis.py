#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np

def load_data(path:str, band:str='raw', order:int=10, shuffle:bool=False):
    """Load data from files.

    Args:
        path (str): folder path of data.
        band (str, optional): name of target band. 'raw' for unfiltered. Defaults to 'raw'.
        order (int, optional): order of regression.
        shuffle (bool, optional): True for loading shuffled dataset. Defaults to False.

    Returns:
        np.ndarray: gc_data and gc_data_shuffle(if shuffle==True).
    """
    gc_data = np.load(path + f'gc_values_{band:s}_order_{order:d}.npy', allow_pickle=True)
    if shuffle:
        gc_data_shuffle = np.load(path + f'gc_values_{band:s}_shuffled_order_{order:d}.npy', allow_pickle=True)

    if shuffle:
        return gc_data, gc_data_shuffle
    else:
        return gc_data

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import ROC_curve, AUC, Youden_Index

    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    multiplicity = data_package['multiplicity']
    stride = data_package['stride']

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

    is_interarea = False  # is inter area or not
    order = 10

    for band in filter_pool:
        # load data for target band
        gc_data, gc_data_shuffle = load_data(path, band, order, shuffle=True)
        # gc_data = load_data(path, band, order, shuffle=False)
        gc_data_flatten = gc_data[~np.eye(stride[-1], dtype=bool)]
        gc_data_flatten[gc_data_flatten<=0] = 1e-5

        # prepare weight_flatten
        weight = data_package['weight']
        weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]

        # setup interarea mask
        if is_interarea:
            interarea_mask = (weight_flatten != 1.5)
            weight_flatten = weight_flatten[interarea_mask]
            log_gc_data = np.log10(gc_data_flatten[interarea_mask])
        else:
            log_gc_data = np.log10(gc_data_flatten)
        log_gc_range = [log_gc_data.min(), log_gc_data.max()]

        # calculate histogram
        (counts, edges) = np.histogram(log_gc_data, bins=100, density=True)

        # create figure canvas
        fig, ax = plt.subplots(2,4,figsize=(20,10))

        SI_value = gc_data_shuffle[~np.eye(stride[-1], dtype=bool)]
        SI_value[SI_value<=0] = 0
        SI_value = SI_value.mean()
        ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
        ax[0,0].axvline(np.log10(SI_value), color='cyan', label='SI')
        # UNCOMMENT to create double Gaussian fitting of GC PDF
        # from scipy.optimize import curve_fit
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

        weight_set = np.unique(weight_flatten)
        log_gc_data_mean = np.array([np.mean(log_gc_data[weight_flatten==key]) for key in weight_set])
        weight_set[weight_set==0]=1e-6
        pval = np.polyfit(np.log10(weight_set), log_gc_data_mean, deg=1)
        ax[1,0].plot(np.log10(weight_flatten+1e-8), log_gc_data.flatten(), 'k.', label='GC samples')
        ax[1,0].plot(np.log10(weight_set), log_gc_data_mean, 'm.', label='GC mean')
        ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
        ax[1,0].set_ylabel(r'$log_{10}\left(GC\right)$')
        ax[1,0].set_xlabel(r'$log_{10}$(Connectivity Strength)')
        ax[1,0].set_title(f'Fitting Slop = {pval[0]:5.3f}')
        ax[1,0].legend(fontsize=15)

        # Draw ROC curves
        threshold_options = [1e-1, 5e-3, 1e-4]
        opt_threshold = np.zeros(len(threshold_options))
        for idx, threshold in enumerate(threshold_options):
            answer = weight_flatten.copy()
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
        if is_interarea:
            plt.savefig(path + f'channel_{band:s}_gc_interarea_order_{order:d}.png')
        else:
            plt.savefig(path + f'channel_{band:s}_gc_order_{order:d}.png')