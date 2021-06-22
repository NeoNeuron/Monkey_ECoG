#! /usr/bin/python 
# Author: Kai Chen

# Make the Figure 3 for paper: Relationship between SC and Residual.

# %%
from utils.core import *
import numpy as np
import matplotlib.pyplot as plt
from MakeFigure1 import axis_log_formater, gen_sc_fc_figure_new
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
# %%
path = 'image/'
data_tdmi = EcogTDMI()
data_tdmi.init_data(path, 'snr_th_kmean_tdmi.pkl')
sc_tdmi, fc_tdmi = data_tdmi.get_sc_fc('ch')
snr_mask_tdmi = data_tdmi.get_snr_mask(path, 'snr_th_kmean_tdmi.pkl')
roi_mask = data_tdmi.roi_mask.copy()

data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
dist_mat = data_package['d_matrix']
d_mat = {band: dist_mat[roi_mask] for band in data_tdmi.filters}

data_gc = EcogGC()
data_gc.init_data()
sc_gc, fc_gc = data_gc.get_sc_fc('ch')
# %%

fig, ax = plt.subplots(1,2,figsize=(9,4), dpi=400)

band = 'raw'
# new_mask = np.ones_like(snr_mask_tdmi[band])
new_mask = snr_mask_tdmi[band].copy()
new_mask[sc_tdmi[band]==0] = False
new_mask[sc_tdmi[band]==1.5] = False

d_metric = 1./d_mat[band]
# regression with distance to get residue
fc_tdmi_res = fc_tdmi[band].copy()
nan_mask = ~np.isnan(np.log10(fc_tdmi_res))
pval = np.polyfit(d_metric[nan_mask*new_mask], np.log10(fc_tdmi_res[nan_mask*new_mask]), deg=1)
fc_tdmi_res[nan_mask*new_mask] = np.abs(fc_tdmi_res[nan_mask*new_mask] - 10**np.polyval(pval, d_metric[nan_mask*new_mask]))

gen_sc_fc_figure_new(ax[0], fc_tdmi_res, sc_tdmi[band], new_mask,)


new_mask = np.ones_like(new_mask, dtype=bool)
new_mask[sc_gc[band]==0] = False
new_mask[sc_gc[band]==1.5] = False
fc_gc_res = fc_gc[band].copy()
nan_mask = ~np.isnan(np.log10(fc_gc_res))
pval = np.polyfit(d_metric[nan_mask], np.log10(fc_gc_res[nan_mask]), deg=1)
fc_gc_res[nan_mask] = np.abs(fc_gc_res[nan_mask] - 10**np.polyval(pval, d_metric[nan_mask]))
gen_sc_fc_figure_new(ax[1], fc_gc_res, sc_gc[band], new_mask)

for axi, labeli in zip(ax, ('TDMI', 'GC')):
    axi.set_title(axi.get_title().replace(band, labeli))
    axi.set_ylabel('Residual')
fig.suptitle(band)

fig.savefig(path+'Figure_3.png')
# %%
def Linear_R2(x:np.ndarray, y:np.ndarray, z:np.ndarray, pval:np.ndarray)->float:
    """Compute R-square value for linear fitting.

    Args:
        x (np.ndarray): variable 1 of function
        y (np.ndarray): variable 2 of function
        y (np.ndarray): image of function
        pval (np.ndarray): parameter of linear fitting

    Returns:
        float: R square value
    """
    mask = ~np.isnan(x)*~np.isnan(y)*~np.isnan(z)*~np.isinf(x)*~np.isinf(y)*~np.isinf(z)# filter out nan
    z_predict = x[mask]*pval[0]+y[mask]*pval[1] + pval[2]
    R = np.corrcoef(z[mask], z_predict)[0,1]
    return R**2

@axis_log_formater(axis='both')
def gen_sc_fc_figure_3d(ax, fc:np.ndarray, 
                        sc:np.ndarray,
                        dist:np.ndarray,
                        snr_mask:np.ndarray=None,
                        is_log:bool=True,
)->plt.Figure:
    """Generated figure for analysis of causal distributions.

    Args:
        tdmi_flatten (np.ndarray): flattened data for target tdmi statistics.
        sc (np.ndarray): flattened data for true connectome.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    if snr_mask is None:
        snr_mask = np.ones_like(fc).astype(bool)
    if is_log:
        log_fc = np.log10(fc)
    else:
        log_fc = fc.copy()

    log_fc[~snr_mask] = np.nan
    # pval = np.polyfit(np.log10(sc+1e-6)[snr_mask], log_fc[snr_mask], deg=1)
    coef_mat = np.vstack((np.log10(sc+1e-6)[snr_mask], np.log10(dist)[snr_mask])).T 
    coef_mat = np.hstack((coef_mat, np.ones((coef_mat.shape[0], 1)))) # adding bias
    b = log_fc[snr_mask]
    pval,_,_,_ = np.linalg.lstsq( coef_mat, b )
    # ax.plot3D(np.log10(sc+1e-6), np.log10(dist), log_fc.flatten(), 'k.', label='TDMI samples')
    ax.scatter3D(np.log10(sc+1e-6), np.log10(dist), log_fc, c='k', s=2, label='TDMI (above SNR th)', zorder=0)
    x_range = (ax.get_xticks()[0], ax.get_xticks()[-1])
    y_range = (ax.get_yticks()[0], ax.get_yticks()[-1])
    z_range = (ax.get_zticks()[0], ax.get_zticks()[-1])
    xx, yy = np.meshgrid(np.linspace(*x_range, 30),np.linspace(*y_range, 30))
    ax.plot_surface(xx, yy, xx * pval[0] + yy * pval[1] + pval[2], 
        alpha=.5, cmap=plt.cm.coolwarm, 
        edgecolors='royalblue', lw=0.,antialiased=False, 
        zorder=0,
    )
    # ax.set_xlabel(r'$log_{10}$(Connectivity Strength)')
    # sc_buffer = sc.copy()
    # sc_buffer[sc_buffer==0] = np.nan
    # ax.set_xlim(*x_range)
    # ax.set_ylim(*y_range)
    # ax.set_zlim(*z_range)
    ax.set_title('r = %5.3f' % 
        Linear_R2(np.log10(sc+1e-6), np.log10(dist), log_fc, pval)**0.5,
        fontsize=16,
    )
    return ax


# %%
fig = plt.figure(figsize=(14,6), dpi=400)
gs = fig.add_gridspec(nrows=1, ncols=2, 
                        left=0.05, right=0.96, top=0.96, bottom=0.05, 
                        wspace=0.15, hspace=0.20)
ax = [fig.add_subplot(i, projection='3d', azim=-20, elev=30) for i in gs] 

band = 'raw'
# new_mask = np.ones_like(snr_mask_tdmi[band])
new_mask = snr_mask_tdmi[band].copy()
new_mask[sc_tdmi[band]==0] = False
new_mask[sc_tdmi[band]==1.5] = False

d_metric = 1./d_mat[band]
# regression with distance to get residue

gen_sc_fc_figure_3d(ax[0], fc_tdmi[band], sc_tdmi[band], d_metric, new_mask,)

new_mask = np.ones_like(new_mask, dtype=bool)
new_mask[sc_gc[band]==0] = False
new_mask[sc_gc[band]==1.5] = False
gen_sc_fc_figure_3d(ax[1], fc_gc[band], sc_gc[band], d_metric, new_mask,)

for axi, labeli in zip(ax, ('TDMI', 'GC')):
    axi.set_title(labeli + ' : ' + axi.get_title())
    axi.set_xlabel('SC')
    axi.set_ylabel('1/Distance')
    axi.set_zlabel(r'$\log_{10}$('+labeli+')', rotation=270)
    axi.xaxis.set_major_locator(MultipleLocator(4))
    axi.xaxis.set_minor_locator(MultipleLocator(2))
    axi.yaxis.set_major_locator(MultipleLocator(1))
    axi.yaxis.set_minor_locator(MultipleLocator(0.5))
    axi.zaxis.set_major_locator(MultipleLocator(1))
fig.suptitle(band)

fig.savefig(path+'Figure_3-1.png')
# %%

# %%
