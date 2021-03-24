#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import time
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.size']=15
plt.rcParams['axes.labelsize'] = 15
from utils.tdmi import Extract_MI_CG
from utils.plot import gen_mi_s_figure
from utils.utils import print_log
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
                'tdmi_mode': 'max',
                }
parser = ArgumentParser(prog='CG_mi_s',
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

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# create adj_weight_flatten by excluding 
#   auto-tdmi in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[cg_mask]
adj_weight_flatten = {band:adj_weight_flatten for band in filter_pool}

tdmi_data = np.load('data/tdmi_data.npz', allow_pickle=True)
tdmi_data_flatten = {}
for band in filter_pool:
    if band in tdmi_data.keys():
        # load data for target band
        tdmi_data_cg = Extract_MI_CG(tdmi_data[band], args.tdmi_mode, stride)
        tdmi_data_flatten[band] = tdmi_data_cg[cg_mask]
    else:
        tdmi_data_flatten[band] = None
    
fig = gen_mi_s_figure(tdmi_data_flatten, adj_weight_flatten)

# edit axis labels
if args.tdmi_mode == 'sum':
    [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\sum TDMI\right)$') for i in (0,4)]
elif args.tdmi_mode == 'max':
    [fig.get_axes()[i].set_ylabel(r'$log_{10}\left(\max (TDMI)\right)$') for i in (0,4)]
[fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
plt.tight_layout()

fname = f'cg_mi-s_{args.tdmi_mode:s}.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)