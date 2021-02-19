import os
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
from GC import auto_reg

def compute_AIC_BIC(data_series):
    order_max = 50
    AIC = np.zeros((data_series.shape[1], order_max))
    BIC = np.zeros_like(AIC)
    rank = np.zeros((data_series.shape[1], order_max))
    for i in range(data_series.shape[1]):
        x = data_series[:,i].copy()
        for j, order in enumerate(np.arange(AIC.shape[1])+1):
            res, rank[i,j] = auto_reg(x, order)
            buffer = (x.shape[0]-order)*np.log(res.std())
            AIC[i,j] = 2*(order + buffer)
            BIC[i,j] = np.log(x.shape[0]-order)*order + 2*buffer
    return AIC, BIC, rank

def get_AICs_BICs(path, force_compute=False):
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    fname_aic = 'gc_aic.npz'
    fname_bic = 'gc_bic.npz'
    fname_rank = 'gc_rank.npz'
    if os.path.isfile(path+fname_aic) \
        and os.path.isfile(path+fname_bic) \
        and os.path.isfile(path+fname_rank) \
        and not force_compute:
        AICs = np.load(path + fname_aic, allow_pickle=True)
        BICs = np.load(path + fname_bic, allow_pickle=True)
        ranks = np.load(path + fname_rank, allow_pickle=True)
    else:
        AICs = {}
        BICs = {}
        ranks = {}
        data_series = data_package['data_series_raw']
        AICs['raw'], BICs['raw'], ranks['raw'] = compute_AIC_BIC(data_series)
        filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
        for band in filter_pool:
            data_series = data_package[f'data_series_{band:s}']
            AICs[band], BICs[band], ranks[band] = compute_AIC_BIC(data_series)
        np.savez(path + fname_aic, **AICs)
        np.savez(path + fname_bic, **BICs)
        np.savez(path + fname_rank, **ranks)
    return AICs, BICs, ranks

def gen_figure(XICs, ranks, title):
    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    # plot XIC figure
    fig = plt.figure(figsize=(14,6), dpi=200)
    # plot raw
    gs_raw = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.25,
                                    top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_raw[0])

    axt = ax.twinx()
    axt.plot(np.arange(51), ls='--', color='g')
    for xic, rank in zip(XICs['raw'], ranks['raw']):
        xic = (xic-xic.min())/(xic.max()-xic.min())
        ax.plot(np.arange(50)+1, xic, 'navy', alpha=.5)
        axt.plot(np.arange(50)+1, rank, 'orange', alpha=.5)
    ax.set_xlabel('order', fontsize=16)
    ax.set_ylabel(f'Normalized {title:s} value', fontsize=16)
    ax.yaxis.label.set_color('navy')
    ax.set_title(f'{title:s} curves of raw', fontsize=16)
    ax.tick_params(axis='y', colors='navy', which='both')
    axt.set_ylabel('Rank of A', fontsize=16)
    axt.yaxis.label.set_color('orange')
    axt.tick_params(axis='y', colors='orange', which='both')
    axt.spines['right'].set_color('orange')

    gs = fig.add_gridspec(nrows=2, ncols=3, 
                        left=0.30, right=0.98, top=0.92, bottom=0.08, 
                        wspace=0.18)
    axs = np.array([fig.add_subplot(i) for i in gs])
    for band, ax in zip(filter_pool, axs):
        axt = ax.twinx()
        axt.plot(np.arange(51), ls='--', color='g')
        for xic, rank in zip(XICs[band], ranks[band]):
            # xic = (xic-xic.min())/(xic.max()-xic.min())
            ax.plot(np.arange(50)+1, xic, 'navy', alpha=.5)
            ax.tick_params(axis='y', color='navy', labelcolor=(1,1,1,0), which='both') # make ticklabel invisible
            ax.spines['left'].set_color('navy')
            axt.plot(np.arange(50)+1, rank, 'orange', alpha=.5)
            axt.tick_params(axis='y', color='orange', labelcolor=(1,1,1,0), which='both') # make ticklabel invisible
            axt.spines['right'].set_color('orange')
        ax.set_title(band, fontsize=16)

    # [axs[i].set_ylabel('Normalized XIC value') for i in (0,3)]
    [axs[i].set_xlabel('order', fontsize=16) for i in (3,4,5)]
    plt.savefig(path+f'gc_{title.lower():s}.png')
    plt.close()

if __name__ == '__main__':
    path = 'data_preprocessing_46_region/'
    AICs, BICs, ranks = get_AICs_BICs(path, False)
    gen_figure(AICs, ranks, 'AIC')
    gen_figure(BICs, ranks, 'BIC')
