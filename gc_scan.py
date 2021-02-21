# Author: Kai Chen
import numpy as np
from GC import GC
from utils.utils import print_log
import time
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
               'order': 6,
               'shuffle': False,
               }
parser = ArgumentParser(prog='gc_scan',
                        description = "Scan pair-wise Granger Causality.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', default=arg_default['path'], nargs='?',
                    type = str, 
                    help = "path of working directory."
                    )
parser.add_argument('order', default=arg_default['order'], nargs='?',
                    type = int, 
                    help = "order of regression model in GC."
                    )
parser.add_argument('shuffle', default=arg_default['shuffle'], nargs='?',
                    type = bool, 
                    help = "Shuffle flag of GC."
                    )
args = parser.parse_args()

data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
stride = data_package['stride']

filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
N = stride[-1]

start = time.time()
gc_total = {}
for band in filters:
    data_series = data_package['data_series_'+band]
    # shuffle data
    if args.shuffle:
        for i in range(data_series.shape[1]):
            np.random.shuffle(data_series[:,i])
    gc_value = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gc_value[i,j] = GC(data_series[:,i],data_series[:,j], args.order)

    if args.shuffle:
        print_log(f'{band:s} shuffled GC (order {args.order:d}) finished', start)
        np.save(args.path + f'gc_{band:s}_shuffled_order_{args.order:d}.npy', gc_value)
    else:
        print_log(f'{band:s} GC (order {args.order:d}) finished', start)
        np.save(args.path + f'gc_{band:s}_order_{args.order:d}.npy', gc_value)
    
    gc_total[band] = gc_value

# unify all data files
if args.shuffle:
    np.savez(args.path + f'gc_shuffled_order_{args.order:d}.npz', **gc_total)
else:
    np.savez(args.path + f'gc_order_{args.order:d}.npz', **gc_total)
# remove temp data files
for band in filters:
    if args.shuffle:
        os.remove(args.path + f'gc_{band:s}_shuffled_order_{args.order:d}.npy')
        print_log(f'Delete {band:s} shuffled GC (order {args.order:d}) finished', start)
    else:
        os.remove(args.path + f'gc_{band:s}_order_{args.order:d}.npy')
        print_log(f'Delete {band:s} GC (order {args.order:d}) finished', start)
