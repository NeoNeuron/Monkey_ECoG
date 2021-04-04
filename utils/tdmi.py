import numpy as np

def MI_stats(tdmi_data:np.ndarray, mi_mode:str)->np.ndarray:
    """Calculate the statistics of MI from TDMI data series.

    Args:
        tdmi_data (np.ndarray): (n, n, n_delay) original TDMI data.
        mi_mode (str): type of statistics, 'max' or 'sum'.

    Raises:
        RuntimeError: Invalid mi_mode.

    Returns:
        np.ndarray: (n,n) target statistics of MI.
    """
    if mi_mode == 'sum':
        return tdmi_data[:,:,1:11].sum(2)
    elif mi_mode == 'max':
        return tdmi_data.max(2)
    else:
        raise RuntimeError('Invalid mi mode.')

def compute_tdmi_full(tdmi_data:np.ndarray):
    # complete the tdmi series
    n_channel = tdmi_data.shape[0]
    n_delay = tdmi_data.shape[2]
    tdmi_data_full = np.zeros((n_channel, n_channel, n_delay*2-1))
    tdmi_data_full[:,:,n_delay-1:] = tdmi_data
    tdmi_data_full[:,:,:n_delay] = np.flip(tdmi_data.transpose([1,0,2]), axis=2)
    return tdmi_data_full

def compute_delay_matrix(tdmi_data:np.ndarray):
    # tdmi_data_full = compute_tdmi_full(tdmi_data)
    # n_delay = tdmi_data.shape[2]
    delay_mat = np.argmax(tdmi_data, axis=2)
    return delay_mat

def compute_snr_matrix(tdmi_data:np.ndarray):
    noise_mat = compute_noise_matrix(tdmi_data)
    # tdmi_data_full = compute_tdmi_full(tdmi_data)
    snr_mat = (tdmi_data.max(2) - noise_mat)/tdmi_data[:,:,2000:].std(2)
    return snr_mat

def compute_noise_matrix(tdmi_data:np.ndarray):
    return tdmi_data[:,:,2000:].mean(2)

def get_sparsity_threshold(mat, p=0.1):
    counts, edges = np.histogram(mat.flatten(), bins=100)
    mid_tick = edges[:-1] + edges[1]-edges[0]
    th_id = np.argmin(np.abs(np.cumsum(counts)/np.sum(counts) + p - 1))
    th_val = mid_tick[th_id]
    return th_val