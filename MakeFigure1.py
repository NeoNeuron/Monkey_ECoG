#! /usr/bin/python 
# Author: Kai Chen

# Make the Figure 1 for paper: Relationship between SC and FC(TDMI).

# %%
from fcpy.core import *
from fcpy.utils import Linear_R2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import functools

def axis_log_formater(axis='y'):
    def formater_single_type(func):
        @functools.wraps(func)
        def formated_plot(*args, **kwargs):
            ax = func(*args, **kwargs)
            # Get the appropriate axis
            if axis == 'y' or axis == 'both':
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
                ticks = ax.get_yticks()
                labels = [('%5.2f'%item).strip(' ').rstrip('0').rstrip('.') for item in ticks]
                labels = [r'10$^{%s}$'%item for item in labels]
                ax.set_yticklabels(labels, fontdict={
                    'ha':'right',
                    'va':'center_baseline',
                })
            if axis == 'x' or axis == 'both':
                ax.xaxis.set_major_locator(MultipleLocator(2))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ticks = ax.get_xticks()
                labels = [('%5.2f'%item).strip(' ').rstrip('0').rstrip('.') for item in ticks]
                labels = [r'10$^{%s}$'%item for item in labels]
                ax.set_xticklabels(labels)
            return ax
        return formated_plot
    return formater_single_type

def spines_formater(func):
    @functools.wraps(func)
    def formated_plot(*args, **kwargs):
        ax = func(*args, **kwargs)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return ax
    return formated_plot
# %%
@spines_formater
@axis_log_formater(axis='both')
def gen_sc_fc_figure_new(ax, fc:np.ndarray, sc:np.ndarray,
    snr_mask:np.ndarray=None, is_log:bool=True, c=None,
    **kwargs
)->plt.Figure:

    if is_log is True or is_log=='y':
        log_fc = np.log10(fc)
        ax.set_ylabel(r'$log_{10}$(TDMI)')
    else:
        log_fc = fc.copy()
        ax.set_ylabel(r'TDMI')
    if is_log is True or is_log=='x':
        log_sc = np.log10(sc+1e-6)
    else:
        log_sc = sc.copy()

    if snr_mask is None:
        snr_mask = np.ones_like(fc).astype(bool)

    log_fc[~snr_mask] = np.nan
    if isinstance(c, np.ndarray):
        vmax = np.abs(c[snr_mask]).max()
        cax = ax.scatter(log_sc[snr_mask], log_fc[snr_mask], c=c[snr_mask], cmap=plt.cm.RdBu_r, vmax=vmax, vmin=-vmax, lw=0, s=15, label='TDMI (above SNR th)', )
        plt.colorbar(cax, ax=ax)
    else:
        ax.scatter(log_sc[snr_mask], log_fc[snr_mask], c=c, lw=0, s=15, alpha=.5, label='TDMI (above SNR th)', )
    ax.set_xlabel('Structure Connectivity')

    # exclude few things for r-value calculation
    log_sc[sc==0] = np.nan
    log_sc[sc==1.5] = np.nan

    x_range = ax.get_xticks()
    sc_edges = np.linspace(x_range[0], x_range[-1], num = 10)
    sc_center = (sc_edges[1:] + sc_edges[:-1])/2
    # average data
    log_fc_mean = np.zeros(len(sc_edges)-1)
    log_sc_xerr = np.zeros(len(sc_edges)-1)
    log_fc_yerr = np.zeros(len(sc_edges)-1)
    for i in range(len(sc_edges)-1):
        mask = (log_sc >= sc_edges[i]) & (log_sc < sc_edges[i+1])
        if mask.sum() == 0:
            log_fc_mean[i] = np.nan
            log_sc_xerr[i] = np.nan
            log_fc_yerr[i] = np.nan
        else:
            log_fc_mean[i] = np.nanmean(log_fc[mask])
            log_sc_xerr[i] = np.nanstd(log_sc[mask])
            log_fc_yerr[i] = np.nanstd(log_fc[mask])
    ax.errorbar(sc_center, log_fc_mean, xerr=log_sc_xerr, yerr=log_fc_yerr, 
        ls='None', marker='s', color='k', markersize=10, zorder=10) #, label='TDMI mean')
    # linear fitting 
    # fit raw data points
    # pval1 = np.polyfit(log_sc[snr_mask], log_fc[snr_mask], deg=1)
    # ax.plot(np.linspace(-5, 0, 10), np.polyval(pval1, np.linspace(-5,0, 10)), 'r', label='Linear Fitting', alpha=.8)
    # fit binned data points
    pval = np.polyfit(sc_center[~np.isnan(log_fc_mean)], log_fc_mean[~np.isnan(log_fc_mean)], deg=1)
    ax.plot(sc_center, np.polyval(pval, sc_center), 'r', label='Linear Fitting')
    ax.set_title('r = %6.3f'%
        Linear_R2(log_sc, log_fc, pval)**0.5,
        fontsize=16,
    )
    plt.draw()
    return ax
# %%

if __name__ == '__main__':
    # %%
    path = 'image/'
    data_tdmi = EcogTDMI()
    data_tdmi.init_data(path, 'snr_th_kmean_tdmi.pkl')
    sc_tdmi, fc_tdmi = data_tdmi.get_sc_fc('ch')
    snr_mask_tdmi = data_tdmi.get_snr_mask(path, 'snr_th_kmean_tdmi.pkl')

    data_gc = EcogGC()
    data_gc.init_data()
    sc_gc, fc_gc = data_gc.get_sc_fc('ch')
    # %%
    fig, ax = plt.subplots(1,2,figsize=(9,4), dpi=400)

    band = 'raw'
    # new_mask = np.ones_like(snr_mask_tdmi[band])
    new_mask = snr_mask_tdmi[band].copy()
    # new_mask[sc_tdmi[band]==0] = False
    # new_mask[sc_tdmi[band]==1.5] = False
    gen_sc_fc_figure_new(ax[0], fc_tdmi[band], sc_tdmi[band], new_mask,)
    gen_sc_fc_figure_new(ax[1], fc_gc[band], sc_gc[band], new_mask,)

    for axi, labeli in zip(ax, ('TDMI', 'GC')):
        axi.set_title(labeli+' : '+axi.get_title())
        axi.set_xlim(-5.5, 0)
        axi.set_ylabel(labeli)
    fig.suptitle(band)

    fig.savefig(path+'Figure_1.png')
    # %%