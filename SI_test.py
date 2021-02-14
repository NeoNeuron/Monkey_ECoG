#!/Users/kchen/miniconda3/bin/python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              accumulative tdmi with order p. By default, p=10.

import numpy as np
import multiprocessing
import time

if __name__ == '__main__':
  from mutual_info_cy import mutual_info
  from tdmi_scan_v2 import print_log
  t0 = time.time()
  # load data
  path = 'data_preprocessing_46_region/'
  data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

  multiplicity = data_package['multiplicity']
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
    fname = f'{key:s}_tdmi_shuffle.npy'
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
    np.save(path + fname, mi_data)

  for band in filter_pool:
    ScanTDMI(band)
    print_log(f"Finish processing {band:s} data", t0)
  finish = time.time()
  print_log(f'totally time cost {finish-t0:5.2f} s.', t0)