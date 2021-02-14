#!/Users/kchen/miniconda3/bin/python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              accumulative tdmi with order p. By default, p=10.

import time

def print_log(string, t0):
  """Print log info.

  Args:
      string (str): string-like information to print.
      t0 (float): time stamp for starting of program.
  """
  print(f"[INFO] {time.time()-t0:6.2f}: {string:s}")

if __name__ == '__main__':
  import numpy as np
  import multiprocessing
  from mutual_info_cy import tdmi as TDMI
  # load data
  path = 'data_preprocessing_46_region/'
  data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)

  multiplicity = data_package['multiplicity']
  stride = data_package['stride']

  #channel index
  def ScanTDMI(band:str='raw', delay_len:int=10, pn:int=None)->None:
    """Scan channel-wise TDMI for target band.

    Args:
        band (str, optional): band to scan. Defaults to 'raw'.
        delay_len (int, optional): number of time delay to be scanned. Defaults to 10.
        pn (int, optional): number of processes for multiprocessing. Defaults to None.
    """
    N = stride[-1]
    mi_data = np.zeros((N, N, delay_len))
    key = 'data_series_'+band
    fname = f'{key:s}_tdmi_total.npy'
    for i in range(N):
      p = multiprocessing.Pool(pn)
      result = [p.apply_async(func = TDMI,
                              args=(data_package[key][:,i],
                                    data_package[key][:,j], 
                                    delay_len)
                              ) for j in range(N)]
      p.close()
      p.join()
      j = 0
      for res in result:
        mi_data[i,j] = res.get()
        j += 1
    np.save(path + fname, mi_data)

  start = time.time()
  filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
  for band in filter_pool:
    ScanTDMI(band, 41)
    print_log(f"Finish processing {band:s} data", start)
  finish = time.time()
  print_log(f'totally time cost {finish-start:5.2f} s.', start)