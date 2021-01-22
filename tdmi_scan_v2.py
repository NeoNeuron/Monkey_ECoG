#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              accumulative tdmi with order p. By default, p=10.

import numpy as np
import multiprocessing
import time
from tdmi_scan import TDMI

if __name__ == '__main__':
  # load data
  path = 'data_preprocessing_46_region/'
  data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

  multiplicity = data_package['multiplicity']

  #channel index
  def ScanTDMI(band:str=None, pn:int=None)->None:
    idx = np.arange(46)
    time_delay = np.arange(0,41)
    mi_data = np.zeros((len(idx), len(idx)), dtype=np.ndarray)
    if band is None:  # using original time series
      key = 'data_series'
    else:
      key = 'data_series_'+band
    fname = f'{key:s}_tdmi_total.npy'
    for i in range(len(idx)):
      for j in range(len(idx)):
        mi_data[i,j] = np.zeros((multiplicity[i], multiplicity[j], len(time_delay)))
        for k in range(multiplicity[i]):
          p = multiprocessing.Pool(pn)
          result = [p.apply_async(func = TDMI,
                                  args=(data_package[key][i][:,k],
                                        data_package[key][j][:,l], 
                                        time_delay
                                        )
                                  ) for l in range(multiplicity[j])]
          p.close()
          p.join()
          l = 0
          for res in result:
            mi_data[i,j][k,l] = res.get()
            l += 1
    np.save(path + fname, mi_data)

  start = time.time()
  filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
  for band in filter_pool:
    ScanTDMI(band)
  finish = time.time()
  print('[-] totally cost %3.3f s.' % (finish - start))
