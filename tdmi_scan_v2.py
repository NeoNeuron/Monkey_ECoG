#!/Users/kchen/miniconda3/bin/python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              maximum tdmi.

import numpy as np
from multiprocessing import Pool

def ScanTDMI(data_series:np.ndarray, delay_len:int=10, pn:int=None)->np.ndarray:
  """Scan channel-wise TDMI for target band.

  Args:
      data_series (np.ndarray): data series. Each column is a time series. 
      delay_len (int, optional): number of time delay to be scanned. Defaults to 10.
      pn (int, optional): number of processes for multiprocessing. Defaults to None.

  Returns:
      np.ndarray: 3D array of tdmi data.
  """
  N = data_series.shape[1]
  mi_data = np.zeros((N, N, delay_len))
  for i in range(N):
    p = Pool(pn)
    result = [
      p.apply_async(
        func = TDMI, args=(data_series[:,i], data_series[:,j], delay_len)
      ) for j in range(N)
    ]
    p.close()
    p.join()
    j = 0
    for res in result:
      mi_data[i,j] = res.get()
      j += 1
  return mi_data

if __name__ == '__main__':
  import time
  from mutual_info_cy import tdmi as TDMI
  from utils.utils import print_log
  import os
  from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
  arg_default = {'path': 'data_preprocessing_46_region/'}
  parser = ArgumentParser(
    prog='tdmi_scan',
    description = "Scan pair-wise time delayed mutual information",
    formatter_class=ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    'path', default=arg_default['path'],
    nargs='?', type = str, 
    help = "path of working directory."
    )
  args = parser.parse_args()
  # load data
  data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
  stride = data_package['stride']

  start = time.time()
  filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
  tdmi_data = {}
  for band in filter_pool:
    tdmi_data[band] = ScanTDMI(data_package['data_series_'+band], 41)
    # save result to temp file.
    fname = f'tdmi_{band:s}.npy'
    np.save(args.path + fname, tdmi_data[band])
    print_log(f"Finish processing {band:s} data, temp data save to {args.path+fname:s}.", start)

  # unify all data files
  fname = 'tdmi_data.npz'
  np.savez(args.path + fname, **tdmi_data)
  print_log(f'Pickled data save to {args.path+fname:s}.', start)
  # remove temp data files
  for band in filter_pool:
    os.remove(args.path + f'tdmi_{band:s}.npy')
    print_log(f'Delete {band:s} TDMI temp data.', start)
  print_log(f'TDMI scan finished.', start)