#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold, TDCC version.

import time
import pickle
import numpy as np
import matplotlib as mpl 
mpl.rcParams['font.size']=20
mpl.rcParams['axes.labelsize']=25
from fcpy.core import EcogTDCC
from fcpy.roc import scan_auc_threshold
from fcpy.plot import gen_auc_threshold_figure
from fcpy.utils import print_log
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'tdmi_snr_analysis/',
                'is_interarea': False,
                }
parser = ArgumentParser(prog='tdcc_auc_threshold',
                        description = "Generate figure for analysis of causality.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', default=arg_default['path'], nargs='?',
                    type = str, 
                    help = "path of working directory."
                    )
parser.add_argument('is_interarea', default=arg_default['is_interarea'], nargs='?', 
                    type=bool, 
                    help = "inter-area flag."
                    )
args = parser.parse_args()

start = time.time()
# Load SC and FC data
# ==================================================
data = EcogTDCC()
data.init_data(args.path, 'snr_th_gauss_tdcc.pkl')
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
    aucs[band], opt_threshold[band] = scan_auc_threshold(fc[band], sc[band], w_thresholds, is_log=False)
    
fig = gen_auc_threshold_figure(aucs, w_thresholds)


if args.is_interarea:
    fname = f'ch_auc-threshold_tdcc_interarea.png'
else:
    fname = f'ch_auc-threshold_tdcc.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)

if args.is_interarea:
    with open(args.path+f'ch_aucs_tdcc_interarea.pkl', 'wb') as f:
        pickle.dump(aucs, f)
    print_log(f'Figure save to {args.path:s}ch_aucs_tdcc_interarea.pkl', start)
else:
    with open(args.path+f'ch_aucs_tdcc.pkl', 'wb') as f:
        pickle.dump(aucs, f)
    print_log(f'Figure save to {args.path:s}ch_aucs_tdcc.pkl', start)