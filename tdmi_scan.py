#!/usr/bin python
# coding: utf-8
# Author: Kai Chen
# Institute: INS, SJTU
# Description: Parallelly calculating TDMI from filtered ECoG data. Return an matrix of
#              accumulative tdmi with order p. By default, p=10.

from mutual_info import mutual_info
import numpy as np
import multiprocessing
import time

# define time-delayed wrapper of mutual_info function
def DMI(x, y, delay):
  if delay == 0:
    data = np.vstack((x,y)).T
  elif delay > 0:
    data = np.vstack((x[:-delay],y[delay:])).T
  elif delay < 0:
    data = np.vstack((x[-delay:],y[:delay])).T
  return mutual_info(data)

def TDMI(x, y, time_range):
  return np.array([DMI(x,y,delay) for delay in time_range])

# mutual information estimation for binary series
def mutual_info_01(x, y):
  """mutual information for 0-1 binary time series
  :param x: first series
  :type x: int of ndarray
  :param y: second series
  :type y: int of ndarray
  :return: mutual information

  """
  N = len(x)
  px = np.zeros(2, dtype=int)
  py = np.zeros(2, dtype=int)
  pxy = np.zeros((2,2), dtype=int)
  px[1] = np.sum(x)
  py[1] = np.sum(y)
  px[0] = N - px[1]
  py[0] = N - py[1]
  x_binary = x.astype(bool)
  y_binary = y.astype(bool)
  pxy[0,0] = np.sum(~x_binary * ~y_binary)
  pxy[0,1] = np.sum(~x_binary *  y_binary)
  pxy[1,0] = np.sum( x_binary * ~y_binary)
  pxy[1,1] = np.sum( x_binary *  y_binary)
  px = np.repeat(np.array([px]), 2, axis = 0).T
  py = np.repeat(np.array([py]), 2, axis = 0)
  return np.sum(pxy*np.log(pxy/px/py))/N + np.log(N)

def DMI_01(x, y, bins, delay):
  if delay == 0:
    x_new = x.copy()
    y_new = y.copy()
  elif delay > 0:
    x_new = x[:-delay].copy()
    y_new = y[delay:].copy()
  elif delay < 0:
    x_new = x[-delay:].copy()
    y_new = y[:delay].copy()
  return mutual_info_01(x_new, y_new)

def TDMI_01(x, y, time_range):
  return np.array([DMI_01(x,y,delay) for delay in time_range])

def Dcorr(x, y, delay):
  if delay == 0:
    x_new = x.copy()
    y_new = y.copy()
  elif delay > 0:
    x_new = x[:-delay].copy()
    y_new = y[delay:].copy()
  elif delay < 0:
    x_new = x[-delay:].copy()
    y_new = y[:delay].copy()
  return np.corrcoef(x_new, y_new)[0,1]


# load data

data_package = np.load('preprocessed_data.npz')

#channel index
def ScanTDMI(datafile:str, band:str, pn:int)->None:
  id_x = data_package['chose']
  id_y = np.arange(126)
  time_delay = np.arange(0,10)
  mi_data = np.zeros((len(id_x), len(id_y)))
  for i in range(len(id_x)):
    p = multiprocessing.Pool(pn)
    result = [p.apply_async(func = TDMI,
                            args=(data_package[datafile+'_'+band][:,id_x[i]],
                                  data_package[datafile+'_'+band][:,id_y[j]], 
                                  time_delay
                                  )
                            ) for j in range(len(id_y))]
    p.close()
    p.join()
    j = 0
    for res in result:
      mi_data[i,j] = np.sum(res.get())
      j += 1
  np.save(datafile+'_'+band+'_tdmi.npy', mi_data)


start = time.time()
filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
for band in filter_pool:
  ScanTDMI('data_r', band, 10)
finish = time.time()
print('[-] totally cost %3.3f s.' % (finish - start))
