#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import numpy as np

def Linear_R2(x:np.ndarray, y:np.ndarray, pval:np.ndarray)->float:
    """Compute R-square value for linear fitting.

    Args:
        x (np.ndarray): variable of function
        y (np.ndarray): image of function
        pval (np.ndarray): parameter of linear fitting

    Returns:
        float: R square value
    """
    mask = ~np.isnan(y) # filter out nan
    y_predict = x[mask]*pval[0]+pval[1]
    R = np.corrcoef(y[mask], y_predict)[0,1]
    return R**2

if __name__ == '__main__':
    import matplotlib as mpl 
    mpl.rcParams['font.size']=15
    mpl.rcParams['axes.labelsize'] = 15
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import load_data, MI_stats

    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]

    tdmi_mode = 'sum' # or max
    is_interarea = False  # is inter area or not

    fig, ax = plt.subplots(2,4,figsize=(20,10))
    ax = ax.reshape((8,))
    for idx, band in enumerate(filter_pool):
        # load data for target band
        tdmi_data = load_data(path, band)
        tdmi_data = MI_stats(tdmi_data, tdmi_mode)
        tdmi_data_flatten = tdmi_data[~np.eye(stride[-1], dtype=bool)]

        # setup interarea mask
        weight = data_package['weight']
        weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
        if is_interarea:
            interarea_mask = (weight_flatten != 1.5)
            weight_flatten = weight_flatten[interarea_mask]
            log_tdmi_data = np.log10(tdmi_data_flatten[interarea_mask])
        else:
            log_tdmi_data = np.log10(tdmi_data_flatten)
        log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

        answer = weight_flatten.copy()
        answer[answer==0]=1e-7
        log_answer = np.log10(answer)
        answer_edges = np.linspace(-6, 1, num = 15)
        log_tdmi_data_mean = np.zeros(len(answer_edges)-1)
        for i in range(len(answer_edges)-1):
            mask = (log_answer >= answer_edges[i]) & (log_answer < answer_edges[i+1])
            if mask.sum() == 0:
                log_tdmi_data_mean[i] = np.nan
            else:
                log_tdmi_data_mean[i] = log_tdmi_data[mask].mean()
        pval = np.polyfit(answer_edges[:-1][~np.isnan(log_tdmi_data_mean)], log_tdmi_data_mean[~np.isnan(log_tdmi_data_mean)], deg=1)
        ax[idx].plot(answer_edges[:-1], log_tdmi_data_mean, 'k.', markersize=15, label='TDMI mean')
        ax[idx].plot(answer_edges[:-1], np.polyval(pval, answer_edges[:-1]), 'r', label='Linear Fitting')
        ticks = [-5, -3, -1]
        labels = ['$10^{%d}$'%item for item in ticks]
        ax[idx].set_xticks(ticks)
        ax[idx].set_xticklabels(labels)

        if band is None:
            ax[idx].set_title(f'Origin ($R^2$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval):5.3f})')
        else:
            ax[idx].set_title(f'{band:s} ($R^2$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval):5.3f})')
        ax[idx].legend(fontsize=15)
        ax[idx].grid(ls='--')

    if tdmi_mode == 'sum':
        [ax[i].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$') for i in (0,4)]
    elif tdmi_mode == 'max':
        [ax[i].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$') for i in (0,4)]
    [ax[i].set_xlabel('Weight') for i in (4,5,6)]

    # make last subfigure invisible
    ax[-1].set_visible(False)

    plt.tight_layout()
    if is_interarea:
        plt.savefig(path + f'mi-s_interarea_{tdmi_mode:s}.png')
    else:
        plt.savefig(path + f'mi-s_{tdmi_mode:s}.png')