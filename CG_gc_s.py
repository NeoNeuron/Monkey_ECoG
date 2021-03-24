#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=15
plt.rcParams['axes.labelsize'] = 15
from utils.utils import CG, print_log
from utils.plot import gen_mi_s_figure
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
                'order': 6,
                }
parser = ArgumentParser(prog='CG_mi_s',
                        description = "Generate figure for analysis of causality.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', default=arg_default['path'], nargs='?',
                    type = str, 
                    help = "path of working directory."
                    )
parser.add_argument('order', default=arg_default['order'], nargs='?',
                    type = int,
                    help = "order for regression mode in GC."
                    )
args = parser.parse_args()

start = time.time()
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# create adj_weight_flatten by excluding 
#   auto-gc in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]
adj_weight_flatten = {band:adj_weight_flatten for band in filter_pool}

gc_data = np.load(args.path + f'gc_order_{args.order:d}.npz', allow_pickle=True)
gc_data_flatten = {}
for band in filter_pool:
    if band in gc_data.keys():
        gc_data_cg = CG(gc_data[band], stride)
        gc_data_cg[gc_data_cg<=0] = 1e-5
        gc_data_flatten[band] = gc_data_cg[cg_mask]
    else:
        gc_data_flatten[band] = None

fig = gen_mi_s_figure(gc_data_flatten, adj_weight_flatten)

for ax in fig.get_axes():
    handles, labels = ax.get_legend_handles_labels()
    labels = [item.replace('TDMI', 'GC') for item in labels]
    ax.legend(handles, labels)
[fig.get_axes()[i].set_ylabel(r'$log_{10}\left((GC)\right)$') for i in (0,4)]
[fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
plt.tight_layout()

fname = f'cg_gc-s_{args.order:d}.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)