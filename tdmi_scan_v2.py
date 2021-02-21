#!/Users/kchen/miniconda3/bin/python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              maximum tdmi.

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
  from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
  arg_default = {'path': 'data_preprocessing_46_region/'}
  parser = ArgumentParser(prog='tdmi_scan',
                          description = "Scan pair-wise time\
                                         delayed mutual information",
                          formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('path', default=arg_default['path'], nargs='?',
                      type = str, 
                      help = "path of working directory."
                      )
  args = parser.parse_args()
  # load data
  data_package = np.load(args.path + 'preprocessed_data.npz', allow_pickle=True)
  stride = data_package['stride']

  #channel index
  def ScanTDMI(band:str='raw', delay_len:int=10, pn:int=None)->np.ndarray:
    """Scan channel-wise TDMI for target band.

    Args:
        band (str, optional): band to scan. Defaults to 'raw'.
        delay_len (int, optional): number of time delay to be scanned. Defaults to 10.
        pn (int, optional): number of processes for multiprocessing. Defaults to None.

    Returns:
        np.ndarray: 3D array of tdmi data.
    """
    N = stride[-1]
    mi_data = np.zeros((N, N, delay_len))
    key = 'data_series_'+band
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
    return mi_data

  start = time.time()
  filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
  tdmi_data = {}
  for band in filter_pool:
    tdmi_data[band] = ScanTDMI(band, 41)
    print_log(f"Finish processing {band:s} data", start)
  fname = 'tdmi_data.npz'
  np.savez(args.path + fname, **tdmi_data)
  finish = time.time()
  print_log(f'Pickled data save to {args.path+fname:s}.', start)
  print_log(f'totally time cost {finish-start:5.2f} s.', start)