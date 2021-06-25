#! /usr/bin/python 
# Author: Kai Chen

# Make the Figure 4 for paper: Graph analysis

# %%
from MakeFigure1 import axis_log_formater, spines_formater
from utils.core import *
from utils.utils import Linear_R2
import numpy as np
import matplotlib.pyplot as plt
import functools
import pickle

def linfit_range(x, y, xrange=None):
	if xrange is None:
		mask = np.ones_like(x, dtype=bool)
	else:
		mask = (x >= xrange[0])*(x < xrange[1])
	pval = np.polyfit(x[mask], y[mask], deg=1)
	r2 = Linear_R2(x[mask], y[mask], pval)
	return pval, np.sqrt(r2)

# %%
def exp_formatter(axis='y'):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            ax = func(*args, **kwargs)
            # Change the ticklabel format to scientific format
            ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

            # Get the appropriate axis
            if axis == 'y':
                ax_axis = ax.yaxis
                x_pos = 0.0
                y_pos = 1.0
                horizontalalignment='left'
                verticalalignment='bottom'
            else:
                ax_axis = ax.xaxis
                x_pos = 1.0
                y_pos = -0.05
                horizontalalignment='left'
                verticalalignment='top'

            # Run plt.tight_layout() because otherwise the offset text doesn't update
            plt.tight_layout()
            ##### THIS IS A BUG 
            ##### Well, at least it's sub-optimal because you might not
            ##### want to use tight_layout(). If anyone has a better way of 
            ##### ensuring the offset text is updated appropriately
            ##### please comment!

            # Get the offset value
            offset = ax_axis.get_offset_text().get_text()

            if len(offset) > 0:
                # Get that exponent value and change it into latex format
                minus_sign = u'\u2212'
                expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
                offset_text = r'x$\mathregular{10^{%d}}$' %expo

                # Turn off the offset text that's calculated automatically
                ax_axis.offsetText.set_visible(False)

                # Add in a text box at the top of the y axis
                ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment)
            return ax
        return wrapped_func
    return wrapper
# %%
def format_exponent(ax, axis='y'):

    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment='left'
        verticalalignment='bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment='right'
        verticalalignment='top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG 
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of 
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' %expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
               horizontalalignment=horizontalalignment,
               verticalalignment=verticalalignment)
    return ax
# %%
path = 'image/'
data_tdmi = EcogTDMI()
data_tdmi.init_data(path, 'snr_th_kmean_tdmi.pkl')
sc_tdmi, fc_tdmi = data_tdmi.get_sc_fc('ch')
roi_mask = data_tdmi.roi_mask.copy()

data_gc = EcogGC()
data_gc.init_data()
sc_gc, fc_gc = data_gc.get_sc_fc('ch')
# %%
def get_degree(con, roi_mask):
	degree = np.zeros(roi_mask.shape[0])
	con_2d = np.zeros_like(roi_mask, dtype=float)
	con_2d[roi_mask] = con
	con_2d[con_2d<=0] = 0
	degree = con_2d.sum(1)
	return degree

def get_clustering(con, roi_mask):
	clustering_coef = np.zeros(roi_mask.shape[0])
	con_2d = np.zeros_like(roi_mask, dtype=float)
	con_2d[roi_mask] = con
	con_2d[con_2d<=0] = 0
	for i in range(roi_mask.shape[0]):
		mask_i = con_2d[:, i]>0
		buffer = con_2d[mask_i, :][:, mask_i]
		clustering_coef[i] = buffer.sum()/(buffer.shape[0]*(buffer.shape[0]-1.0))
	return clustering_coef
# %%
@spines_formater
@exp_formatter(axis='y')
@exp_formatter(axis='x')
def gen_sc_fc_feature_comp(ax, sc_feature, fc_feature, label='feature'):
    pval, r = linfit_range(sc_feature, fc_feature, )
    x_range = np.linspace(sc_feature.min(), sc_feature.max(), 10)
    ax.plot(sc_feature, fc_feature, '.', color='gray', ms=10)
    ax.plot(x_range, np.polyval(pval, x_range), color='red')
    ax.set_xlabel('Structural %s'%label.capitalize())
    ax.set_ylabel('Functional %s'%label.capitalize())
    ax.set_title(f'r={r:5.3f}')
    return ax
#%%
with open('image/th_gap_tdmi.pkl', 'rb') as f:
    fc_tdmi_th = pickle.load(f)
with open('image/th_gap_gc.pkl', 'rb') as f:
    fc_gc_th = pickle.load(f)

band = 'raw'
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=400)
sc_tmp = sc_tdmi[band].copy()
plot_binary_flag = True

if plot_binary_flag:
    sc_tmp = (sc_tmp > 1e-2).astype(float)
    fc_tdmi_tmp = (fc_tdmi[band]>fc_tdmi_th[band]).astype(float)
    fc_gc_tmp = (fc_gc[band]>fc_gc_th[band]).astype(float)
else:
    sc_tmp[sc_tmp==1.5] = 0
    sc_tmp = sc_tmp/sc_tmp.max()
    fc_tdmi_tmp = fc_tdmi[band]/fc_tdmi_th[band].max()
    fc_gc_tmp = fc_gc[band]/fc_gc_th[band].max()

sc_degree = get_degree(sc_tmp, roi_mask)
fc_degree = get_degree(fc_tdmi_tmp, roi_mask)
gen_sc_fc_feature_comp(ax[0], sc_degree, fc_degree, 'degree')
fc_degree = get_degree(fc_gc_tmp, roi_mask)
gen_sc_fc_feature_comp(ax[1], sc_degree, fc_degree, 'degree')
for i, label in enumerate(('TDMI', 'GC')):
    ax[i].set_title(label+' : '+ax[i].get_title())
    # ax[i].set_xlim(0,7)

fig.savefig(path+'Figure_4-1.png')

#%%
band = 'raw'
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=400)

sc_cluster = get_clustering(sc_tmp, roi_mask)
fc_cluster = get_clustering(fc_tdmi_tmp, roi_mask)
gen_sc_fc_feature_comp(ax[0], sc_cluster, fc_cluster, 'clustering')

fc_cluster = get_clustering(fc_gc_tmp, roi_mask)
gen_sc_fc_feature_comp(ax[1], sc_cluster, fc_cluster, 'clustering')
for i, label in enumerate(('TDMI', 'GC')):
    ax[i].set_title(label+' : '+ax[i].get_title())

fig.savefig(path+'Figure_4-2.png')
# %%
@axis_log_formater(axis='x')
@axis_log_formater(axis='y')
@spines_formater
def gen_con_hist(ax, connectivity, fit_range):
    con = connectivity.copy()
    (counts, edges) = np.histogram(np.log10(con[con!=0]), bins=50)
    counts = (counts+1)/counts.sum()
    con_grid = (edges[1:]+edges[:-1])/2
    ax.plot(con_grid, np.log10(counts), '.', color='navy')
    pval, r = linfit_range(con_grid, np.log10(counts), fit_range)
    x_range = np.linspace(fit_range[0], fit_range[1], 10)
    ax.plot(x_range, np.polyval(pval, x_range), color='k', label=f'k={pval[0]:4.2f}')
    ax.set_title(f'r = {r:5.3f}')
    ax.set_xlabel('Connectivity')
    ax.set_ylabel(r'$P\left(Connectivity\right)$')
    ax.legend()
    return ax
# %%
fig, ax = plt.subplots(1,1, figsize=(5,4), dpi=400)
gen_con_hist(ax, sc_tdmi['raw'], (-3,-1))
fig.savefig(path+'Figure_4-3.png')
# %%
band = 'raw'
fig, ax = plt.subplots(1,2, figsize=(10,4), dpi=400)
gen_con_hist(ax[0], fc_tdmi[band], (-2, -0))
ax[0].set_title('TDMI'+' : '+ax[0].get_title())
ax[0].set_xlabel('TDMI')
ax[0].set_ylabel(r'$P\left(TDMI\right)$')

gen_con_hist(ax[1], fc_gc[band], (-3, -1))
ax[1].set_title('GC'+' : '+ax[1].get_title())
ax[1].set_xlabel(r'GC')
ax[1].set_ylabel(r'$P\left(GC\right)$')
fig.suptitle(band)
fig.savefig(path+'Figure_4-4.png')

# %%
