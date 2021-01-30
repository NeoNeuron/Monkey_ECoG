#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              accumulative tdmi with order p. By default, p=10.

import numpy as np
import multiprocessing
import time
from tdmi_scan import DMI

def print_log(string, t0):
  print(f"[INFO] {time.time()-t0:5.2f}: {string:s}")

if __name__ == '__main__':
  t0 = time.time()
  # load data
  path = 'data_preprocessing_46_region/'
  data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

  multiplicity = data_package['multiplicity']
  
  # shuffle data
  print_log("start shuffling data", t0)
  filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', None]
  for band in filter_pool:
    if band is None:  # using original time series
      key = 'data_series'
    else:
      key = 'data_series_'+band
    for i in range(len(data_package[key])):
      for j in range(data_package[key][i].shape[1]):
        np.random.shuffle(data_package[key][i][:,j])

  print_log("finish shuffling data", t0)

  #channel index
  def ScanTDMI(band:str=None, pn:int=None)->None:
    idx = np.arange(46)
    mi_data = np.zeros((len(idx), len(idx)), dtype=np.ndarray)
    if band is None:  # using original time series
      key = 'data_series'
    else:
      key = 'data_series_'+band
    fname = f'{key:s}_tdmi_shuffle.npy'
    for i in range(len(idx)):
      for j in range(len(idx)):
        mi_data[i,j] = np.zeros((multiplicity[i], multiplicity[j]))
        # auto select the axes with smaller length to loop
        if multiplicity[i] <= multiplicity[j]:
          for k in range(multiplicity[i]):
            p = multiprocessing.Pool(pn)
            result = [p.apply_async(func = DMI,
                                    args=(data_package[key][i][:,k],
                                          data_package[key][j][:,l], 
                                          0
                                          )
                                    ) for l in range(multiplicity[j])]
            p.close()
            p.join()
            l = 0
            for res in result:
              mi_data[i,j][k,l] = res.get()
              l += 1
        else:
          for l in range(multiplicity[j]):
            p = multiprocessing.Pool(pn)
            result = [p.apply_async(func = DMI,
                                    args=(data_package[key][i][:,k],
                                          data_package[key][j][:,l], 
                                          0
                                          )
                                    ) for k in range(multiplicity[i])]
            p.close()
            p.join()
            k = 0
            for res in result:
              mi_data[i,j][k,l] = res.get()
              k += 1
    np.save(path + fname, mi_data)

  start = time.time()
  for band in filter_pool:
    if band is None:
      print_log("start processing raw data", t0)
    else: 
      print_log(f"start processing {band:s} data", t0)
    ScanTDMI(band)
    if band is None:
      print_log("finish processing raw data", t0)
    else:
      print_log(f"finish processing {band:s} data", t0)
  finish = time.time()
  print_log(f'totally time cost {finish-start:3.3f} s.', t0)
