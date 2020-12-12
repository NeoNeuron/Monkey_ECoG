"""Created by Kai Chen

Module for band filters applied on continues LFP series

"""
import numpy as np
from scipy.signal import butter, sosfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs = fs, output = 'sos')
    y = sosfilt(sos, data)
    yr = y[::-1]
    y = sosfilt(sos, yr)
    return y[::-1]

def butter_lowpass_filter(data, highcut, fs, order=5):
    sos = butter(order, highcut, btype='lowpass', fs = fs, output = 'sos')
    y = sosfilt(sos, data)
    yr = y[::-1]
    y = sosfilt(sos, yr)
    return y[::-1]

def filter(data, band, fs, order = 5):
    """
    Signal filter of target band.

    Parameters
    ----------
    data : narray of floats
        Original data series
    band : string
        Band of filter, 
        includeing 'delta', 'theta', 'alpha', 'beta', 'gamma', and 'high_gamma'
    fs : float
        Sampling frequency
    order : int
        Order of filtering funcion. Actual order is doubled.

    Returns
    -------
    y : narray of floats
        Filtered series.

    Example
    -------
    >>> y = filter(data, 'alpha', fs = 2000, order = 5)

    """
    
    if band == 'delta':
        y = butter_bandpass_filter(data, lowcut = 1, highcut = 4, fs = fs, order = order)
    elif band == 'theta':
        y = butter_bandpass_filter(data, lowcut = 5, highcut = 8, fs = fs, order = order)
    elif band == 'alpha':
        y = butter_bandpass_filter(data, lowcut = 9, highcut = 12, fs = fs, order = order)
    elif band == 'beta':
        y = butter_bandpass_filter(data, lowcut = 14, highcut = 30, fs = fs, order = order)
    elif band == 'gamma':
        y = butter_bandpass_filter(data, lowcut = 31, highcut = 100, fs = fs, order = order)
    elif band == 'high_gamma':
        y = butter_bandpass_filter(data, lowcut = 55, highcut = 100, fs = fs, order = order)
    else:
        raise ValueError('Invalid band option')
    return y


def main(prefix, lfp_fname, freq, verbose):
    '''
    Filter LFP data with bandpass filter.

    Parameters
    ----------
    prefix : string
        prefix of input and output data
    lfp_fname : string
        filename of LFP data
    freq : string
        type of bandpass filter
    verbose : bool
        flag to print the runing time

    Return
    ------
    lfp_filtered : array of floats
        filtered LFP data

    '''

    import time
    t_start = time.time()
    lfp = np.load(prefix + lfp_fname)
    lfp_filtered = filter(lfp, fs = 2000, band = freq)
    out_dir = prefix + lfp_fname.split('.')[0] + '-' + freq + '.npy' 
    np.save(out_dir, lfp_filtered)
    t_finish = time.time()
    if verbose:
        print('>> Filtered data output to %s'%out_dir)
        print('>> Filtering took %5.3e s'% (t_finish-t_start))
    return lfp_filtered

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "Frequency filter for LFP signal.")
    parser.add_argument('prefix', type = str, help = 'working directory of source data and output data')
    parser.add_argument('lfp_fname', type = str, help = 'filename of LFP data')
    parser.add_argument('freq', type = str, choices=['alpha', 'beta', 'theta', 'gamma', 'delta', 'ripple', 'lfp'], help = 'frequency band')
    parser.add_argument('-v', '--verbose', action='store_true', help = 'enable verbose of running time')
    args = parser.parse_args()
    main(**(vars(args)))
