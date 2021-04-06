# linear fitting between weight(log) and tdmi value(log).
# pairs with sufficient SNR value are counted.

if __name__ == '__main__':
    from utils.binary_threshold import *
    from utils.core import EcogTDMI
    import numpy as np
    import pickle
    path='tdmi_snr_analysis/'

    suffix = '_tdmi'
    with open(path + 'th_gap'+suffix+'.pkl', 'rb') as f:
        gap_th = pickle.load(f)
    
    gap_th['delta'] = 10**(-0.3)
    with open(path + 'th_gap'+suffix+'.pkl', 'wb') as f:
        pickle.dump(gap_th, f)



    suffix = '_tdmi_CG'
    with open(path + 'th_gap'+suffix+'.pkl', 'rb') as f:
        gap_th = pickle.load(f)
    
    gap_th['delta'] = 10**(-0.32)
    gap_th['beta'] = 10**(-2)
    gap_th['gamma'] = 10**(-3)
    with open(path + 'th_gap'+suffix+'.pkl', 'wb') as f:
        pickle.dump(gap_th, f)