#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold, GC version.

import time
import pickle
import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
from utils.core import EcogGC
from utils.roc import scan_auc_threshold
from utils.plot import gen_auc_threshold_figure
from utils.utils import print_log
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
# Load SC and FC data
# ==================================================
data = EcogGC()
data.init_data()
sc, fc = data.get_sc_fc('ch')
# ==================================================
w_thresholds = np.logspace(-6, 0, num=7, base=10)

aucs = {}
opt_threshold = {}
for band in data.filters:
    if args.is_interarea:
        interarea_mask = (sc[band] != 1.5)
        sc[band] = sc[band][interarea_mask]
        fc[band] = fc[band][interarea_mask]
    aucs[band], opt_threshold[band] = scan_auc_threshold(fc[band], sc[band], w_thresholds)
    
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

if args.is_interarea:
    with open(args.path+f'gc_aucs_interarea_order_{args.order:d}.pkl', 'wb') as f:
        pickle.dump(aucs, f)
    print_log(f'Figure save to {args.path:s}gc_aucs_interarea_order_{args.order:d}.pkl', start)
else:
    with open(args.path+f'gc_aucs_order_{args.order:d}.pkl', 'wb') as f:
        pickle.dump(aucs, f)
    print_log(f'Figure save to {args.path:s}gc_aucs_order_{args.order:d}.pkl', start)