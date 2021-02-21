#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import numpy as np
import matplotlib.pyplot as plt

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

def gen_mi_s_figure(tdmi_data_flatten:dict, weight_flatten:np.ndarray)->plt.Figure:
    fig, ax = plt.subplots(2,4,figsize=(20,10))
    ax = ax.reshape((8,))
    idx = 0
    for band, tdmi_flatten in tdmi_data_flatten.items():
        log_tdmi_data = np.log10(tdmi_flatten)

        answer = weight_flatten.copy()
        answer[answer==0]=1e-7
        log_answer = np.log10(answer)
        answer_edges = np.linspace(-6, 1, num = 15)
        # average data
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
        ax[idx].set_title(f'{band:s} ($r$ = {Linear_R2(answer_edges[:-1], log_tdmi_data_mean, pval)**0.5:5.3f})')
        ax[idx].legend(fontsize=15)
        ax[idx].grid(ls='--')
        idx += 1

    # make last subfigure invisible
    ax[-1].set_visible(False)

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    import time
    plt.rcParams['font.size']=15
    plt.rcParams['axes.labelsize'] = 15
    from draw_causal_distribution_v2 import MI_stats
    from tdmi_scan_v2 import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                    'tdmi_mode': 'max',
                    'is_interarea': False,
                    }
    parser = ArgumentParser(prog='plot_mi_s',
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
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    weight = data_package['weight']
    # setup interarea mask
    weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]
    if args.is_interarea:
        interarea_mask = (weight_flatten != 1.5)
        weight_flatten = weight_flatten[interarea_mask]
    tdmi_data = np.load(args.path + 'tdmi_data.npz', allow_pickle=True)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
    tdmi_data_flatten = {}
    for band in filter_pool:
        tdmi_data_flatten[band] = MI_stats(tdmi_data[band], args.tdmi_mode)
        tdmi_data_flatten[band] = tdmi_data_flatten[band][~np.eye(stride[-1], dtype=bool)]
        if args.is_interarea:
            tdmi_data_flatten[band] = tdmi_data_flatten[band][interarea_mask]

    fig = gen_mi_s_figure(tdmi_data_flatten, weight_flatten)

    # edit axis labels
    if args.tdmi_mode == 'sum':
        [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$') for i in (0,4)]
    elif args.tdmi_mode == 'max':
        [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$') for i in (0,4)]
    [fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
    plt.tight_layout()

    if args.is_interarea:
        fname = f'mi-s_interarea_{args.tdmi_mode:s}.png'
    else:
        fname = f'mi-s_{args.tdmi_mode:s}.png'
    fig.savefig(args.path + fname)
    print_log(f'Figure save to {args.path+fname:s}.', start)