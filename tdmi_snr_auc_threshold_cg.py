#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
#   Coarse grain causal analysis across cortical regions
#   Plot AUC vs. answer threshold.

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=20
plt.rcParams['axes.labelsize']=25
from utils.roc import scan_auc_threshold
from utils.plot import gen_auc_threshold_figure
from utils.tdmi import MI_stats 
from utils.tdmi import compute_snr_matrix, compute_noise_matrix
from utils.utils import CG, print_log
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {
    'path': 'tdmi_snr_analysis/',
    'tdmi_mode': 'max',
    'is_interarea': False,
}
parser = ArgumentParser(
    prog='GC plot_auc_threshold',
    description = "Generate figure for analysis of causality.",
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    'path', 
    default=arg_default['path'], 
    nargs='?',
    type = str, 
    help = "path of working directory."
)
parser.add_argument(
    'tdmi_mode', 
    default=arg_default['tdmi_mode'], 
    nargs='?',
    type = str, choices=['max', 'sum'], 
    help = "TDMI mode."
)
args = parser.parse_args()

start = time.time()
data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

w_thresholds = np.logspace(-6, 0, num=7, base=10)
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# manually set snr threshold
with open(args.path+'snr_th.pkl', 'rb') as f:
    snr_th = pickle.load(f)
# create adj_weight_flatten by excluding 
#   auto-tdmi in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten = adj_weight[~cg_mask]

# load data for target band
tdmi_data = np.load('data/tdmi_data_long.npz', allow_pickle=True)
aucs = {}
aucs_no_snr = {}
for band in filter_pool:
    if band in tdmi_data.keys():
        # generate SNR mask
        snr_mat = compute_snr_matrix(tdmi_data[band])
        noise_matrix = compute_noise_matrix(tdmi_data[band])
        snr_mask = snr_mat >= snr_th[band]
        # compute TDMI statistics
        tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
        tdmi_data_cg_no_snr = CG(tdmi_data_band, stride)
        tdmi_data_band[~snr_mask] = noise_matrix[~snr_mask]
        # compute coarse-grain average
        tdmi_data_cg = CG(tdmi_data_band, stride)
        # apply cg mask
        tdmi_data_flatten = tdmi_data_cg[~cg_mask]
        tdmi_data_flatten_no_snr = tdmi_data_cg_no_snr[~cg_mask]

        aucs_no_snr[band], _ = scan_auc_threshold(tdmi_data_flatten_no_snr, adj_weight_flatten, w_thresholds)
        aucs[band], _= scan_auc_threshold(tdmi_data_flatten, adj_weight_flatten, w_thresholds)
    else:
        aucs_no_snr[band], aucs[band] = None, None

fig = gen_auc_threshold_figure(aucs_no_snr, w_thresholds, labels="No SNR mask")
gen_auc_threshold_figure(aucs, w_thresholds, ax=np.array(fig.get_axes()), colors='orange', labels="SNR mask")
[axi.legend() for axi in fig.get_axes()[:-1]]

fname = f'cg_auc-threshold_{args.tdmi_mode:s}_manual-th.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)
with open(args.path+f'cg_aucs_{args.tdmi_mode:s}.pkl', 'wb') as f:
    pickle.dump(aucs_no_snr, f)
    pickle.dump(aucs, f)
print_log(f'Figure save to {args.path:s}cg_aucs_{args.tdmi_mode:s}.pkl', start)