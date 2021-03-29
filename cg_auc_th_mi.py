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
from utils.core import EcogTDMI
from utils.roc import scan_auc_threshold
from utils.plot import gen_auc_threshold_figure
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
args = parser.parse_args()

start = time.time()
# Load SC and FC data
# ==================================================
data = EcogTDMI('data/')
data.init_data()
sc, fc = data.get_sc_fc('cg')
# ==================================================
w_thresholds = np.logspace(-6, 0, num=7, base=10)

aucs = {}
for band in data.filters:
    aucs[band], _ = scan_auc_threshold(fc[band], sc[band], w_thresholds)

fig = gen_auc_threshold_figure(aucs, w_thresholds)

fname = f'cg_auc-threshold.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)