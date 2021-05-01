#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# TDMI delay analysis. plot the delay statistics.

import matplotlib.pyplot as plt 

def plot_distance_delay(ax, data:dict,):
    delay_mat = data['delay']
    d_matrix = data['dist']
    ax.plot(d_matrix, delay_mat, '.')
    ax.set_ylabel('Delay (ms)')
    ax.set_xlabel('Channel Distance')
    return ax

if __name__ == '__main__':
    import numpy as np 
    from utils.plot import plot_union
    def axis_formattor(ax):
        ax.axis('scaled')
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return ax
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import utils
    arg_default = {'path': 'tdmi_snr_analysis/'}
    parser = ArgumentParser(prog='tdmi_delay_analysis',
                            description = "Scan pair-wise maximum delay and SNR\
                                            of time delayed mutual information",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', default=arg_default['path'], nargs='?',
                        type = str, 
                        help = "path of working directory."
                        )
    args = parser.parse_args()

    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    d_matrix = data_package['d_matrix']
    data = utils.core.EcogTDMI()
    delay_matrix = data.get_delay_matrix()
    snr_mask = data.get_snr_mask(args.path)
    roi_mask = data.compute_roi_masking('ch')

    delay_th = 60
    mask = {}
    for band in data.filters:
        delay_mat = np.abs(delay_matrix[band])
        mask[band] = (delay_mat<delay_th)
        mask[band][roi_mask] *= snr_mask[band]

    data_plt = {}
    for band in data.filters:
        data_plt[band] = {'delay':np.abs(delay_matrix[band][mask[band]]), 'dist':d_matrix[mask[band]]}
    # plot delay matrices
    # -------------------
    fig = plot_union(data_plt, plot_distance_delay)

    plt.savefig(args.path+f'tdmi_delay_dist.png')
    plt.close()