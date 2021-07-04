#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Analyze the causal relation calculated from ECoG data.

import numpy as np
from fcpy.tdmi import MI_stats

def CG(tdmi_data:np.ndarray, stride:np.ndarray)->np.ndarray:
    """Compute the coarse-grained average of 
        each cortical region for tdmi_data.

    Args:
        tdmi_data (np.ndarray): channel-wise tdmi_data.
        stride (np.ndarray): stride of channels. 
            Equal to the `cumsum` of multiplicity.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data
        np.ndarray: dispersion index of coarse-grained averaged tdmi_data 
    """
    multiplicity = np.diff(stride).astype(int)
    n_region = stride.shape[0]-1
    tdmi_data_cg = np.zeros((n_region, n_region))
    cg_dispersion = np.zeros_like(tdmi_data_cg)
    max_mean_ratio = np.ones_like(tdmi_data_cg)
    for i in range(n_region):
        for j in range(n_region):
            data_buffer = tdmi_data[stride[i]:stride[i+1],stride[j]:stride[j+1]]
            if i != j:
                tdmi_data_cg[i,j]=data_buffer.mean()
                cg_dispersion[i,j]=data_buffer.std()/data_buffer.mean()
                max_mean_ratio[i,j]=data_buffer.max()/data_buffer.mean()
            else:
                if multiplicity[i] > 1:
                    tdmi_data_cg[i,j]=np.mean(data_buffer[~np.eye(multiplicity[i], dtype=bool)])
                    cg_dispersion[i,j]=np.std(data_buffer[~np.eye(multiplicity[i], dtype=bool)])/tdmi_data_cg[i,j]
                    max_mean_ratio[i,j]=np.max(data_buffer[~np.eye(multiplicity[i], dtype=bool)])/tdmi_data_cg[i,j]
                else:
                    tdmi_data_cg[i,j]=data_buffer.mean() # won't be used in ROC.
    return tdmi_data_cg, cg_dispersion, max_mean_ratio

def Extract_MI_CG(tdmi_data:np.ndarray, mi_mode:str, stride:np.ndarray)->np.ndarray:
    """Extract coarse-grained tdmi_data from original tdmi data.

    Args:
        tdmi_data (np.ndarray): original tdmi data
        mi_mode (str): mode of mi statistics
        stride (np.ndarray): stride of channels.
            Equal to the `cumsum` of multiplicity.

    Returns:
        np.ndarray: coarse-grained average of tdmi_data.
        np.ndarray: dispersion index of coarse-grained averaged tdmi_data 
    """
    tdmi_data = MI_stats(tdmi_data, mi_mode)
    tdmi_data_cg, cg_dispersion, max_mean_ratio = CG(tdmi_data, stride)
    return tdmi_data_cg, cg_dispersion, max_mean_ratio

if __name__ == '__main__':
    import matplotlib as mpl 
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    import matplotlib.pyplot as plt
    from draw_causal_distribution_v2 import load_data

    path = 'data_preprocessing_46_region/'
    data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
    stride = data_package['stride']
    multiplicity = np.diff(stride).astype(int)
    n_region = multiplicity.shape[0]

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

    tdmi_mode = 'max'  # or 'max'

    # create adj_weight_flatten by excluding 
    #   auto-tdmi in region with single channel
    adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
    cg_mask = ~np.diag(multiplicity == 1).astype(bool)
    adj_weight_flatten = adj_weight[cg_mask]

    for band in filter_pool:
        # load data
        tdmi_data = load_data(path, band)
        _, cg_dispersion, max_mean_ratio = Extract_MI_CG(tdmi_data, tdmi_mode, stride)

        # tdmi_data_flatten = tdmi_data_cg[cg_mask]
        # log_tdmi_data = np.log10(tdmi_data_flatten)
        # log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

        fig, ax = plt.subplots(1,2,figsize=(10,4))
        # calculate histogram
        cg_dispersion = cg_dispersion[cg_mask]
        cg_dispersion = cg_dispersion[cg_dispersion!=0]
        (counts, edges) = np.histogram(cg_dispersion, bins=100)
        ax[0].plot(edges[:-1], counts, '-*', color='navy')
        ax[0].set_xlabel('Standard Deviation-Mean Ratio')
        ax[0].set_ylabel('Counts')

        max_mean_ratio = max_mean_ratio[cg_mask]
        max_mean_ratio = max_mean_ratio[max_mean_ratio!=1]
        (counts, edges) = np.histogram(max_mean_ratio, bins=100)
        ax[1].plot(edges[:-1], counts, '-*', color='navy')
        ax[1].set_xlabel('Max-mean Ratio')
        ax[1].set_ylabel('Counts')
        ax[1].set_xlim(0.95,7.3)

        plt.tight_layout()
        plt.savefig('tmp/' + 'cg_dispersion_'+band+'_'+tdmi_mode+'.png')