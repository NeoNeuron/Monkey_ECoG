#!/Users/kchen/miniconda3/bin/python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from shuffled ECoG data. 

import numpy as np
import multiprocessing
import time

if __name__ == '__main__':
  from mutual_info_cy import mutual_info
  from utils.utils import print_log
  from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
  arg_default = {'path': 'data_preprocessing_46_region/'}
  parser = ArgumentParser(prog='tdmi_scan_significant_test',
                          description = "Scan shuffled mutual information",
                          formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('path', default=arg_default['path'], nargs='?',
                      type = str, 
                      help = "path of working directory."
                      )
  args = parser.parse_args()
  t0 = time.time()
  # load data
  data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
  stride = data_package['stride']
  filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
  data_dict = {}
  
  # shuffle data
  print_log("start shuffling data", t0)
  for band in filter_pool:
    key = 'data_series_'+band
    # copy to shuffle
    data_dict[key] = data_package[key]
    for idx in range(data_dict[key].shape[1]):
      np.random.shuffle(data_dict[key][:,idx])

  print_log("finish shuffling data", t0)

  #channel index
  def ScanTDMI(band:str='raw', pn:int=None)->None:
    key = 'data_series_'+band
    N = stride[-1]
    mi_data = np.zeros((N, N))
    for i in range(N):
      p = multiprocessing.Pool(pn)
      result = [p.apply_async(func = mutual_info,
                              args=(data_dict[key][:,i], data_dict[key][:,j])
                              ) for j in range(N)]
      p.close()
      p.join()
      j = 0
      for res in result:
        mi_data[i,j] = res.get()
        j += 1
    return mi_data

  tdmi_data = {}
  for band in filter_pool:
    tdmi_data[band] = ScanTDMI(band)
    print_log(f"Finish processing {band:s} data", t0)
  fname = 'tdmi_data_shuffle.npz'
  np.savez(args.path + fname, **tdmi_data)
  finish = time.time()
  print_log(f'Pickled data save to {args.path+fname:s}.', t0)
  print_log(f'totally time cost {finish-t0:5.2f} s.', t0)