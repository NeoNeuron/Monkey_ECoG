#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import time
import numpy as np
import matplotlib.pyplot as plt 
plt.rcParams['font.size']=15
plt.rcParams['axes.labelsize'] = 15
from utils.tdmi import MI_stats
from utils.tdmi import compute_snr_matrix
from utils.plot import gen_mi_s_figure
from utils.utils import CG, print_log
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {
    'path': 'tdmi_snr_analysis/',
    'tdmi_mode': 'max',
}
parser = ArgumentParser(prog='CG_mi_s',
    description = "Generate figure for analysis of causality.",
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    'path', default=arg_default['path'], nargs='?',
    type = str, 
    help = "path of working directory."
)
parser.add_argument(
    'tdmi_mode', default=arg_default['tdmi_mode'], nargs='?',
    type = str, choices=['max', 'sum'], 
    help = "TDMI mode."
)
args = parser.parse_args()

start = time.time()
data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']
multiplicity = np.diff(stride).astype(int)

filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']

# manually set snr threshold
snr_th = {
    'raw'        :5.0,
    'delta'      :4.3,
    'theta'      :4.5,
    'alpha'      :4.,
    'beta'       :5.,
    'gamma'      :11,
    'high_gamma' :11,
}

# create adj_weight_flatten by excluding 
#   auto-tdmi in region with single channel
adj_weight = data_package['adj_mat'] + np.eye(data_package['adj_mat'].shape[0])*1.5
cg_mask = ~np.diag(multiplicity == 1).astype(bool)
adj_weight_flatten_template = adj_weight[cg_mask]
adj_weight_flatten = {band:adj_weight_flatten_template for band in filter_pool}

tdmi_data = np.load(args.path + 'tdmi_data.npz', allow_pickle=True)
tdmi_data_flatten = {}
for band in filter_pool:
    if band in tdmi_data.keys():
        # generate SNR mask
        snr_mat = compute_snr_matrix(tdmi_data[band])
            # th_val = get_sparsity_threshold(snr_mat, p = 0.2)
            # snr_mask = snr_mat >= th_val
        snr_mask = snr_mat >= snr_th[band]
        # compute TDMI statistics
        tdmi_data_band = MI_stats(tdmi_data[band], args.tdmi_mode)
        # set filtered entities as numpy.nan
        tdmi_data_band[~snr_mask] = np.nan
        # compute coarse-grain average
        tdmi_data_cg = CG(tdmi_data_band, stride)
        # apply cg mask
        tdmi_data_flatten[band] = tdmi_data_cg[cg_mask]
        # remove potential np.nan entities
        nan_mask = ~np.isnan(tdmi_data_flatten[band])
        # apply nan_mask
        tdmi_data_flatten[band] = tdmi_data_flatten[band][nan_mask]
        adj_weight_flatten[band] = adj_weight_flatten[band][nan_mask]
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

fname = f'cg_mi-s_{args.tdmi_mode:s}_manual-th.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)