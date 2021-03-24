#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm, colors
from draw_causal_distribution_v2 import load_data, MI_stats

path = "data_preprocessing_46_region/"
path_short = "data_preprocessing_46_region_short/"
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']

tdmi_mode = 'max'

band = None
tdmi_data_long  = load_data(path, band)
tdmi_data_long = MI_stats(tdmi_data_long, tdmi_mode)
tdmi_data_long = tdmi_data_long[~np.eye(stride[-1], dtype=bool)]
tdmi_data_short = load_data(path_short, band)
tdmi_data_short = MI_stats(tdmi_data_short, tdmi_mode)
tdmi_data_short = tdmi_data_short[~np.eye(stride[-1], dtype=bool)]
axis_type = 'log'
    
fig, ax = plt.subplots(1,1,figsize=(4,3.2), dpi=300)
if axis_type == 'linear':
    ax.plot(tdmi_data_short, tdmi_data_long, '.k')
elif axis_type == 'log':
    ax.loglog(tdmi_data_short, tdmi_data_long, '.k')
ax.plot(range(2), '--', color='orange')
ax.axis('scaled')
ax.set_xlabel('MI valude (nats) [Data length = 12 seconds]', fontsize=8)
ax.set_ylabel('MI valude (nats) [Data length = 24 seconds]', fontsize=8)
ax.set_title(r'$\max_\tau$TDMI$(\tau)$ vs length of series', fontsize=10)

plt.tight_layout()
plt.grid(ls='--', color='grey', lw=0.5)
if band is None:
    plt.savefig(path_short+f'tdmi_vs_length_{axis_type:s}.png')
else:
    plt.savefig(path_short+f'tdmi_vs_length_{band:s}_{axis_type:s}.png')
plt.close()