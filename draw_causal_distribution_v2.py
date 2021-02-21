#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np
import matplotlib.pyplot as plt

# define fitting function
def Gaussian(x, a, mu, sigma):
    """Standand Gaussian template function.

    Args:
        x (array-like): 1-d input data.
        a (float): positive contrained Gaussian amplitude
        mu (float): mean
        sigma (float): variance

    Returns:
        array-like: function value
    """
    return np.abs(a)*np.exp(-(x-mu)**2/sigma)

def Double_Gaussian(x, a1, a2, mu1, mu2, sigma1, sigma2):
    """Double Gaussian like template function.

    Args:
        x (array-like): 1d input data
        a1 (float): amplitude of first Gaussian
        a2 (float): amplitude of second Gaussian
        mu1 (float): mean of first Gaussian
        mu2 (float): mean of second Gaussian
        sigma1 (float): variance of first Gaussian
        sigma2 (float): variance of second Gaussian

    Returns:
        array-like: function value
    """
    return Gaussian(x, a1, mu1, sigma1) + Gaussian(x, a2, mu2, sigma2)

def MI_stats(tdmi_data:np.ndarray, mi_mode:str)->np.ndarray:
    """Calculate the statistics of MI from TDMI data series.

    Args:
        tdmi_data (np.ndarray): (n, n, n_delay) original TDMI data.
        mi_mode (str): type of statistics, 'max' or 'sum'.

    Raises:
        RuntimeError: Invalid mi_mode.

    Returns:
        np.ndarray: (n,n) target statistics of MI.
    """
    if mi_mode == 'sum':
        return tdmi_data[:,:,1:11].sum(2)
    elif mi_mode == 'max':
        return tdmi_data.max(2)
    else:
        raise RuntimeError('Invalid mi mode.')

def ROC_curve(y_true:np.ndarray, y_score:np.ndarray, thresholds:np.ndarray):
    """Compute Receiver Operating Characteristic(ROC) curve.

    Args:
        y_true (np.ndarray): binary 1d array for true category labels.
        y_score (np.ndarray): 1d array of score, can be probability measure.
        thresholds (np.ndarray): array of thresholds for binary classification.

    Raises:
        TypeError: non boolean type element in y_true.

    Returns:
        np.ndarray : false_positive
            false positive rate.
        np.ndarray : true_positive
            true positive rate.
    """
    if y_true.dtype != bool:
        raise TypeError('y_true.dtype should be boolean.')
    false_positive = np.array([np.sum((y_score>threshold)*(~y_true))/np.sum(~y_true)
                     for threshold in thresholds])
    true_positive  = np.array([np.sum((y_score>threshold)*(y_true))/np.sum(y_true)
                     for threshold in thresholds])
    return false_positive, true_positive

def Youden_Index(fpr:np.ndarray, tpr:np.ndarray)->int:
    """Compute Youden's Statistics(Youden Index) of ROC curve.

    Args:
        fpr (np.ndarray): false positive rate(specificity)
        tpr (np.ndarray): true positive rate(sensitivity)

    Returns:
        int: Youden index
    """
    y = tpr - fpr
    return np.argmax(y)  # Only the first occurrence is returned.

def AUC(fpr:np.ndarray, tpr:np.ndarray)->float:
    """Calculate AUC of ROC_curve. Numerical scheme: Trapezoid Rule.

    Args:
        fpr (np.ndarray): false positive rate
        tpr (np.ndarray): true positive rate

    Returns:
        float: area under the curve
    """
    return -np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2)

def load_data(path:str, band:str='raw', shuffle:bool=False):
    """Load data from files.

    Args:
        path (str): folder path of data.
        band (str, optional): name of target band. 'raw' for unfiltered. Defaults to 'raw'.
        shuffle (bool, optional): True for loading shuffled dataset. Defaults to False.

    Returns:
        np.ndarray: tdmi_data and tdmi_data_shuffle(if shuffle==True).
    """
    tdmi_data = np.load(path + 'tdmi_data.npz', allow_pickle=True)[band]
    if shuffle:
        tdmi_data_shuffle = np.load(path + 'tdmi_data_shuffle.npz', allow_pickle=True)[band]

    if shuffle:
        return tdmi_data, tdmi_data_shuffle
    else:
        return tdmi_data

def gen_causal_distribution_figure(tdmi_flatten:np.ndarray, 
                                   weight_flatten:np.ndarray,
                                   tdmi_threshold:float, 
                                   )->plt.Figure:
    """Generated figure for analysis of causal distributions.

    Args:
        tdmi_flatten (np.ndarray): flattened data for target tdmi statistics.
        weight_flatten (np.ndarray): flattened data for true connectome.
        tdmi_threshold (float): significance value of tdmi statistics.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    log_tdmi_data = np.log10(tdmi_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # calculate histogram
    (counts, edges) = np.histogram(log_tdmi_data, bins=100, density=True)

    # create figure canvas
    fig, ax = plt.subplots(2,4,figsize=(20,10))

    ax[0,0].plot(edges[1:], counts, '-*k', label='Raw')
    ax[0,0].axvline(np.log10(tdmi_threshold), color='cyan', label='SI')
    # UNCOMMENT to create double Gaussian fitting of TDMI PDF
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
    log_tdmi_data_mean = np.array([np.mean(log_tdmi_data[weight_flatten==key]) for key in weight_set])
    weight_set[weight_set==0]=1e-6
    pval = np.polyfit(np.log10(weight_set), log_tdmi_data_mean, deg=1)
    ax[1,0].plot(np.log10(weight_flatten+1e-8), log_tdmi_data.flatten(), 'k.', label='TDMI samples')
    ax[1,0].plot(np.log10(weight_set), log_tdmi_data_mean, 'm.', label='TDMI mean')
    ax[1,0].plot(np.log10(weight_set), np.polyval(pval, np.log10(weight_set)), 'r', label='Linear Fitting')
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
    return fig

if __name__ == '__main__':
    import time
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from tdmi_scan_v2 import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'tdmi_mode': 'max',
                   'is_interarea': False,
                   'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
                   }
    parser = ArgumentParser(prog='draw_causal_distribution',
                            description = "Generate figure for analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
                        type = str, choices=['max', 'sum'], 
                        help = "TDMI mode."
                        )
    parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                        type=bool, 
                        help = "inter-area flag."
                        )
    parser.add_argument('--filters', default=arg_default['filters'], nargs='*', 
                        type=str, 
                        help = "list of filtering band."
                        )
    args = parser.parse_args()


    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    # prepare weight_flatten
    weight = data_package['weight']
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]

    for band in args.filters:
        # load data for target band
        tdmi_data, tdmi_data_shuffle = load_data(args.path, band, shuffle=True)

        tdmi_data = MI_stats(tdmi_data, args.tdmi_mode)
        tdmi_data_flatten = tdmi_data[~np.eye(stride[-1], dtype=bool)]

        # setup interarea mask
        if args.is_interarea:
            tdmi_data_flatten = tdmi_data_flatten[interarea_mask]

        SI_value = tdmi_data_shuffle[~np.eye(stride[-1], dtype=bool)].mean()
        if args.tdmi_mode == 'sum':
            SI_value *= 10
        fig = gen_causal_distribution_figure(tdmi_data_flatten, 
                                             weight_flatten,
                                             SI_value)

        if args.tdmi_mode == 'sum':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
        elif args.tdmi_mode == 'max':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        if args.is_interarea:
            fname = f'channel_{band:s}_interarea_analysis_{args.tdmi_mode:s}.png'
        else:
            fname = f'channel_{band:s}_analysis_{args.tdmi_mode:s}.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)