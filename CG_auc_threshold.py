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
from utils.roc import scan_auc_threshold
from utils.plot import gen_auc_threshold_figure
from utils.tdmi import Extract_MI_CG
from utils.utils import print_log
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
                'tdmi_mode': 'max',
                'is_interarea': False,
                }
parser = ArgumentParser(prog='GC plot_auc_threshold',
                        description = "Generate figure for analysis of causality.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', default=arg_default['path'], nargs='?',
                    type = str, 
                    help = "path of working directory."
                    )
parser.add_argument('tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
                    type = str, choices=['max', 'sum'], 
                    help = "TDMI mode."
                    )
args = parser.parse_args()

start = time.time()
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

w_thresholds = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# create adj_weight_flatten by excluding 
#   auto-tdmi in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]

# load data for target band
tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
aucs = {}
opt_threshold = {}
for band in filter_pool:
    if band in tdmi_data.keys():
        tdmi_data_flatten = Extract_MI_CG(tdmi_data[band], args.tdmi_mode, stride)
        tdmi_data_flatten = tdmi_data_flatten[cg_mask]
        aucs[band], opt_threshold[band] = scan_auc_threshold(tdmi_data_flatten, adj_weight_flatten, w_thresholds)
    else:
        aucs[band], opt_threshold[band] = None, None

fig = gen_auc_threshold_figure(aucs, w_thresholds)

# save optimal threshold computed by Youden Index
np.savez(args.path + f'opt_threshold_cg_tdmi_{args.tdmi_mode:s}.npz', **opt_threshold)

fname = f'cg_auc-threshold_{args.tdmi_mode:s}.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)