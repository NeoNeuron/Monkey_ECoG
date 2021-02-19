import os
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
from GC import create_structure_array

def compute_AIC(data_series):
    AIC = np.zeros((data_series.shape[1], 50))
    for i in range(AIC.shape[0]):
        x = data_series[:,i].copy()
        for j, order in enumerate(np.arange(AIC.shape[1])+1):
            x_array = create_structure_array(x, order)
            # y_array = create_structure_array(y, order)
            auto_coeff = np.linalg.lstsq(x_array, x[order:],rcond=-1)[0]
            res = x[order:]-x_array @ auto_coeff
            AIC[i,j] = 2*(order + (x.shape[0]-order)*np.log(res.std()))
    return AIC

def get_AICs(path, force_compute=False):
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    fname = 'gc_aic.npz'
    if os.path.isfile(path+fname) and not force_compute:
        AICs = np.load(path + fname, allow_pickle=True)
    else:
        AICs = {}
        data_series = data_package['data_series_raw']
        AICs['raw'] = compute_AIC(data_series)
        filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
        for band in filter_pool:
            data_series = data_package[f'data_series_{band:s}']
            AICs[band] = compute_AIC(data_series)
    return AICs

if __name__ == '__main__':
    path = 'data_preprocessing_46_region/'
    AICs = get_AICs(path, False)

    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.25,
                                    top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])

    for aic in AICs['raw']:
        aic = (aic-aic.min())/(aic.max()-aic.min())
        ax.plot(np.arange(50)+1, aic, 'navy', alpha=.5)
    ax.set_xlabel('order')
    ax.set_ylabel('Normalized AIC value')
    ax.set_title('raw')

    gs = fig.add_gridspec(nrows=2, ncols=3, 
                        left=0.30, right=0.98, top=0.92, bottom=0.08, 
                        wspace=0.15)
    axs = np.array([fig.add_subplot(i) for i in gs])
    for band, ax in zip(filter_pool, axs):
        for aic in AICs[band]:
            aic = (aic-aic.min())/(aic.max()-aic.min())
            ax.plot(np.arange(50)+1, aic, 'navy', alpha=.5)
        ax.set_title(band)

    [axs[i].set_ylabel('Normalized AIC value') for i in (0,3)]
    [axs[i].set_xlabel('order', fontsize=16) for i in (3,4,5)]
    plt.savefig(path+'gc_aic.png')
    plt.close()




