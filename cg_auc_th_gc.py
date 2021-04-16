#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
#   Coarse grain causal analysis across cortical regions
#   Plot AUC vs. answer threshold.

import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams['font.size']=20
plt.rcParams['axes.labelsize']=25
from utils.core import EcogGC
from utils.utils import print_log
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
# Load SC and FC data
# ==================================================
data = EcogGC()
data.init_data()
sc, fc = data.get_sc_fc('cg')
# ==================================================

w_thresholds = np.logspace(-6, 0, num=7, base=10)
aucs = {}
opt_threshold = {}
for band in data.filters:
    if fc[band] is not None:
        aucs[band], opt_threshold[band] = scan_auc_threshold(fc[band], sc[band], w_thresholds)
    else:
        aucs[band], opt_threshold[band] = None, None
        print_log(f"Data for {band:s} band does not exist.", start)

fig = gen_auc_threshold_figure(aucs, w_thresholds)

fname = f'cg_auc-threshold_gc_{args.order:d}.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)
plt.savefig(args.path + f'cg_auc-threshold_gc_{args.order:d}.png')
with open(args.path+f'cg_aucs_gc_order_{args.order:d}.pkl', 'wb') as f:
    pickle.dump(aucs, f)
print_log(f'Figure save to {args.path:s}cg_aucs_gc_order_{args.order:d}.pkl', start)