#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
#   Coarse grain causal analysis across cortical regions
#   Plot AUC vs. answer threshold.

import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=20
plt.rcParams['axes.labelsize']=25
from utils.utils import CG, print_log
from utils.roc import scan_auc_threshold
from utils.plot import gen_auc_threshold_figure
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
                'order': 6,
                }
parser = ArgumentParser(prog='CG_gc_auc_threshold',
                        description = "Generate figure for analysis of causality.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', default=arg_default['path'], nargs='?',
                    type = str, 
                    help = "path of working directory."
                    )
parser.add_argument('order', default=arg_default['order'], nargs='?',
                    type = int,
                    help = "order of regression model in GC."
                    )
args = parser.parse_args()

start = time.time()
data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# create adj_weight_flatten by excluding 
#   auto-gc in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]
# load gc_data
gc_data = np.load(args.path + f'gc_order_{args.order:d}.npz', allow_pickle=True)

aucs = {}
opt_threshold = {}
for band in filter_pool:
    if band in gc_data.keys():
        gc_data_cg = CG(gc_data[band], stride)
        gc_data_flatten = gc_data_cg[cg_mask]
        gc_data_flatten[gc_data_flatten<=0] = 1e-5
        aucs[band], opt_threshold[band] = scan_auc_threshold(gc_data_flatten, adj_weight_flatten, w_thresholds)
    else:
        aucs[band], opt_threshold[band] = None, None

fig = gen_auc_threshold_figure(aucs, w_thresholds)

# save optimal threshold computed by Youden Index
np.savez(args.path + f'opt_threshold_cg_gc_order_{args.order:d}.npz', **opt_threshold)

fname = f'cg_auc-threshold_gc_{args.order:d}.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)
plt.savefig(args.path + f'cg_auc-threshold_gc_{args.order:d}.png')