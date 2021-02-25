import numpy as np 
import matplotlib.pyplot as plt
# plt.rcParams['axes.linewidth']=0.5
plt.rcParams['lines.linewidth']=0.1
from utils.tdmi import compute_tdmi_full, compute_delay_matrix
from utils.tdmi import compute_snr_matrix
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

path = 'data_preprocessing_46_region/'
band = 'raw'
tdmi_data = np.load(path+'tdmi_data.npz', allow_pickle=True)
tdmi_data_full = compute_tdmi_full(tdmi_data[band])
delay_mat = compute_delay_matrix(tdmi_data[band]) 
snr_mat = compute_snr_matrix(tdmi_data[band])
snr_mask = snr_mat >= 5
print(f'Percentage filtered : {snr_mask.sum()/snr_mat.shape[0]**2}:5.2f %')
n_delay = tdmi_data[band].shape[2]
fig = plt.figure(constrained_layout=False, figsize=(16,12))
gs = fig.add_gridspec(nrows=3, ncols=1, 
                        left=0.10, right=0.98, top=0.98, bottom=0.55, 
                        wspace=0.01)
ax = [fig.add_subplot(i) for i in gs]
gs = fig.add_gridspec(nrows=1, ncols=1, 
                        left=0.10, right=0.98, top=0.52, bottom=0.05, 
                        wspace=0.01)
ax.append(fig.add_subplot(gs[0]))

ax0_mask = snr_mask*(delay_mat>200)
for curve, delay in zip(tdmi_data_full[ax0_mask], delay_mat[ax0_mask]):
    ax[0].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.8, color='r') 
    ax[0].axvline(delay, alpha=.6, color='y', ls='--') 
ax[0].set_title(r'delay $\in$ (200, 1000] ms')
ax[0].set_ylim(0,)
ax[0].set_xlim(-1000,1000)
ax[0].xaxis.set_ticklabels([])

ax1_mask = snr_mask*(delay_mat<-200)
for curve, delay in zip(tdmi_data_full[ax1_mask], delay_mat[ax1_mask]):
    ax[1].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.8, color='b') 
    ax[1].axvline(delay, alpha=.6, color='y', ls='--') 
ax[1].set_title(r'delay $\in$ [-1000, -200) ms')
ax[1].set_ylim(0,)
ax[1].set_xlim(-1000,1000)
ax[1].xaxis.set_ticklabels([])

[ax[2].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='g') 
    for curve in tdmi_data_full[snr_mask*(delay_mat==0)]]
ax[2].set_title(r'delay = 0 ms')
ax[2].set_xlim(-1000,1000)
ax[2].set_ylim(0,)
ax[2].xaxis.set_ticklabels([])

ax3_mask = snr_mask*(delay_mat!=0)*(delay_mat>=-200)*(delay_mat<=200)
axins = zoomed_inset_axes(ax[3], zoom=4, loc='upper right')
for curve, delay in zip(tdmi_data_full[ax3_mask], delay_mat[ax3_mask]):
    ax[3].plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='grey') 
    axins.plot(np.arange(2*n_delay-1)-n_delay+1, curve, alpha=.6, color='grey') 
    ax[3].axvline(delay, alpha=.6, color='y', ls='--', lw=0.2) 
    axins.axvline(delay, alpha=.6, color='y', ls='--', lw=0.2) 
ax[3].set_title(r'delay $\in$ [-200, 0) $\cup$ (0, 200] ms')
ax[3].set_xlabel('Time delay (ms)', fontsize=16)
ax[3].set_xlim(-1000,1000)
ax[3].set_ylim(0,)
ax3_ylim = ax[3].get_ylim()
print(ax3_ylim)
axins.set_xlim(-100,100)
axins.set_ylim(ax3_ylim[0]+ax3_ylim[1]/3,ax3_ylim[1]/8+ax3_ylim[1]/3)
# fix the number of ticks on the inset axes
# axins.yaxis.get_major_locator().set_params(nbins=7)
# axins.xaxis.get_major_locator().set_params(nbins=7)
# plt.setp(axins.get_xticklabels(), visible=False)
plt.setp(axins.get_yticklabels(), visible=False)
mark_inset(ax[3], axins, loc1=2, loc2=4, fc="none", ec="0.5", ls='--')

textstr = 'Mutual Information (nats)'
props = dict(boxstyle=None, facecolor='w', edgecolor='w', alpha=0.)
ax[3].text(-0.08, 1.50, textstr, transform=ax[3].transAxes, fontsize=30,
        verticalalignment='top', bbox=props, rotation='vertical', )

plt.savefig('tmp/delay_analysis_tmp.png')
plt.close()
