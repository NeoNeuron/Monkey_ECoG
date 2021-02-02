#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np
from draw_causal_distribution_v2 import MI_stats

def CG(tdmi_data:np.ndarray, stride:np.ndarray, multiplicity:np.ndarray=None)->np.ndarray:
    """Compute the coarse-grained average of 
        each cortical region for tdmi_data.

    Args:
        tdmi_data (np.ndarray): channel-wise tdmi_data.
        stride (np.ndarray): stride of channels. 
            Equal to the `cumsum` of multiplicity.
        multiplicity (np.ndarray, optional): number of channel
            in each cortical region. Defaults to None.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data
    """
    if multiplicity is None:
        multiplicity = np.diff(stride).astype(int)
    n_region = stride.shape[0]-1
    tdmi_data_cg = np.zeros((n_region, n_region))
    for i in range(n_region):
        for j in range(n_region):
            data_buffer = tdmi_data[stride[i]:stride[i+1],stride[j]:stride[j+1]]
            if i != j:
                tdmi_data_cg[i,j]=data_buffer.mean()
            else:
                if multiplicity[i] > 1:
                    tdmi_data_cg[i,j]=np.mean(data_buffer[~np.eye(multiplicity[i], dtype=bool)])
                else:
                    tdmi_data_cg[i,j]=data_buffer.mean() # won't be used in ROC.
    return tdmi_data_cg

def Extract_MI_CG(tdmi_data:np.ndarray, mi_mode:str, stride:np.ndarray, 
                  multiplicity:np.ndarray=None)->np.ndarray:
    """Extract coarse-grained tdmi_data from original tdmi data.

    Args:
        tdmi_data (np.ndarray): original tdmi data
        mi_mode (str): mode of mi statistics
        stride (np.ndarray): stride of channels.
            Equal to the `cumsum` of multiplicity.
        multiplicity (np.ndarray): number of channels
            in each cortical region. Default to None.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data.
    """
    tdmi_data = MI_stats(tdmi_data, mi_mode)
    tdmi_data_cg = CG(tdmi_data, stride, multiplicity)
    return tdmi_data_cg

if __name__ == '__main__':
    import matplotlib as mpl 
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import load_data, ROC_curve, AUC, Youden_Index

    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    multiplicity = data_package['multiplicity']
    stride = data_package['stride']
    n_region = multiplicity.shape[0]

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

    tdmi_mode = 'sum'  # or 'max'

    # create adj_weight_flatten by excluding 
    #   auto-tdmi in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    for band in filter_pool:
        # load shuffled tdmi data for target band
        tdmi_data = load_data(path, band, shuffle=True)
        tdmi_data_cg = Extract_MI_CG(tdmi_data, tdmi_mode, stride, multiplicity)

        tdmi_data_flatten = tdmi_data_cg[cg_mask]
        log_tdmi_data = np.log10(tdmi_data_flatten)
        log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

        # calculate histogram
        (counts, edges) = np.histogram(log_tdmi_data, bins=100, density=True)

        fig, ax = plt.subplots(2,4,figsize=(20,10))

        SI_value = tdmi_data_shuffle[~np.eye(stride[-1], dtype=bool)].mean()
        if tdmi_mode == 'sum':
            SI_value *= 10
        ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
        ax[0,0].axvline(np.log10(SI_value), color='cyan', label='SI')
        # UNCOMMENT to create double Gaussian fitting of TDMI PDF
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
        log_tdmi_data_mean = np.array([np.mean(log_tdmi_data[adj_weight_flatten==key]) for key in weight_set])
        weight_set[weight_set==0]=1e-6
        pval = np.polyfit(np.log10(weight_set), log_tdmi_data_mean, deg=1)
        ax[1,0].plot(np.log10(adj_weight_flatten+1e-8), log_tdmi_data, 'k.', label='TDMI samples')
        # ax[1,0].plot(np.log10(weight_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
        ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
        if tdmi_mode == 'sum':
            ax[1,0].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
        elif tdmi_mode == 'max':
            ax[1,0].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
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
            thresholds = np.linspace(log_tdmi_range[0],log_tdmi_range[1],100)
            fpr, tpr = ROC_curve(answer, log_tdmi_data, thresholds)
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
        if band == None:
            plt.savefig(path + 'cg_analysis_'+tdmi_mode+'.png')
        else:
            plt.savefig(path + 'cg_'+band+'_analysis_'+tdmi_mode+'.png')