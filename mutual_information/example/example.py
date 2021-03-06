import numpy as np
from minfo.mi_float import mutual_info as mi_cy
from minfo.mi_float import tdmi as TDMI_cy
from minfo.mi_float import tdmi_omp as TDMI_cy_omp
from mutual_info import mutual_info as mi_py
import time

def TDMI_py(dat, n):
    tdmi = np.zeros(n)
    N = dat.shape[0]
    for i in range(n):
        dat_buffer = np.zeros((N-i, 2))
        dat_buffer[:,0] = dat[:N-i,0]
        dat_buffer[:,1] = dat[i:,1]
        tdmi[i] = mi_py(dat_buffer)
    return tdmi

n = 24000
dat = np.zeros((n,2))
dat[:,0] = np.random.rand(n)
dat[:,1] = dat[:,0]
# np.random.shuffle(dat[:,1])
n_delay = 100
t0 = time.time()
mi_cy(dat[:,0], dat[:,1])
print(f'[INFO]:   mi (cython)        takes {time.time() - t0:3.3f} s')

t0 = time.time()
TDMI_cy(dat[:,0], dat[:,1], n_delay)
print(f'[INFO]: tdmi (cython)        takes {time.time() - t0:3.3f} s')

t0 = time.time()
TDMI_cy_omp(dat[:,0], dat[:,1], n_delay)
print(f'[INFO]: tdmi (cython/OpenMP) takes {time.time() - t0:3.3f} s')

t0 = time.time()
mi_py(dat)
print(f'[INFO]:   mi (python)        takes {time.time() - t0:3.3f} s')

t0 = time.time()
TDMI_py(dat, n_delay)
print(f'[INFO]: tdmi (python)        takes {time.time() - t0:3.3f} s')