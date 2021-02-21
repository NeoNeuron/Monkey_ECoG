#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np
from draw_causal_distribution_v2 import MI_stats

def CG(tdmi_data:np.ndarray, stride:np.ndarray)->np.ndarray:
    """Compute the coarse-grained average of 
        each cortical region for tdmi_data.

    Args:
        tdmi_data (np.ndarray): channel-wise tdmi_data.
        stride (np.ndarray): stride of channels. 
            Equal to the `cumsum` of multiplicity.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data
    """
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

def Extract_MI_CG(tdmi_data:np.ndarray, mi_mode:str, stride:np.ndarray)->np.ndarray:
    """Extract coarse-grained tdmi_data from original tdmi data.

    Args:
        tdmi_data (np.ndarray): original tdmi data
        mi_mode (str): mode of mi statistics
        stride (np.ndarray): stride of channels.
            Equal to the `cumsum` of multiplicity.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data.
    """
    tdmi_data = MI_stats(tdmi_data, mi_mode)
    tdmi_data_cg = CG(tdmi_data, stride)
    return tdmi_data_cg

if __name__ == '__main__':
    import time
    import matplotlib as mpl 
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import load_data, gen_causal_distribution_figure
    from tdmi_scan_v2 import print_log
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    arg_default = {'path': 'data_preprocessing_46_region/',
                   'tdmi_mode': 'max',
                   'filters': ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw'],
                   }
    parser = ArgumentParser(prog='CG_causal_distribution',
                            description = "Generate figure for coarse-grain analysis of causality.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    parser.add_argument('tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
                        type = str, choices=['max', 'sum'], 
                        help = "TDMI mode."
                        )
    parser.add_argument('--filters', default=arg_default['filters'], nargs='*', 
                        type=str, 
                        help = "list of filtering band."
                        )
    args = parser.parse_args()

    start = time.time()
    data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    n_region = multiplicity.shape[0]

    # create adj_weight_flatten by excluding 
    #   auto-tdmi in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    for band in args.filters:
        # load shuffled tdmi data for target band
        tdmi_data, tdmi_data_shuffle = load_data(args.path, band, shuffle=True)
        tdmi_data_cg = Extract_MI_CG(tdmi_data, args.tdmi_mode, stride)

        tdmi_data_flatten = tdmi_data_cg[cg_mask]

        SI_value = tdmi_data_shuffle[~np.eye(stride[-1], dtype=bool)].mean()
        if args.tdmi_mode == 'sum':
            SI_value *= 10

        fig = gen_causal_distribution_figure(tdmi_data_flatten, 
                                             adj_weight_flatten,
                                             SI_value,
                                             )

        if args.tdmi_mode == 'sum':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$')
        elif args.tdmi_mode == 'max':
            fig.get_axes()[4].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$')
        fig.get_axes()[4].get_lines()[1].set_color((0,0,0,0))
        handles, labels = fig.get_axes()[4].get_legend_handles_labels()
        fig.get_axes()[4].legend((handles[0],handles[2]), (labels[0], labels[2]))
        plt.tight_layout()
        print_log(f"Figure {band:s} generated.", start)

        fname = f'cg_{band:s}_analysis_{args.tdmi_mode:s}.png'
        fig.savefig(args.path + fname)
        print_log(f'Figure save to {args.path+fname:s}.', start)
        plt.close(fig)