#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold, GC version.

import time
import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
import matplotlib.pyplot as plt
from plot_auc_threshold import scan_auc_threshold, gen_auc_threshold_figure
from tdmi_scan_v2 import print_log
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
                'order': 6,
                'is_interarea': False,
                }
parser = ArgumentParser(prog='gc_auc_threshold',
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
parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                    type=bool, 
                    help = "inter-area flag."
                    )
args = parser.parse_args()

start = time.time()
data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
weight = data_package['weight']
weight_flatten = weight[~np.eye(stride[-1], dtype=bool)]

# setup interarea mask
if args.is_interarea:
    interarea_mask = (weight_flatten != 1.5)
    weight_flatten = weight_flatten[interarea_mask]
w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
gc_data = np.load(args.path + f'gc_order_{args.order:d}.npz', allow_pickle=True)

aucs = {}
opt_threshold = {}
for band in filter_pool:
    # load data for target band
    gc_data_flatten = gc_data[band][~np.eye(stride[-1], dtype=bool)]
    gc_data_flatten[gc_data_flatten<=0] = 1e-5
    if args.is_interarea:
        gc_data_flatten = gc_data_flatten[interarea_mask]

    aucs[band], opt_threshold[band] = scan_auc_threshold(gc_data_flatten, weight_flatten, w_thresholds)
    
fig = gen_auc_threshold_figure(aucs, w_thresholds)

# save optimal threshold computed by Youden Index
if args.is_interarea:
    np.savez(args.path + f'opt_threshold_channel_interarea_gc_order_{args.order:d}.npz', **opt_threshold)
else:
    np.savez(args.path + f'opt_threshold_channel_gc_order_{args.order:d}.npz', **opt_threshold)

if args.is_interarea:
    fname = f'gc_auc-threshold_interarea_{args.order:d}.png'
else:
    fname = f'gc_auc-threshold_{args.order:d}.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)
