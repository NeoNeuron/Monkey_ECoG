# Author: Kai Chen
from .tdmi import MI_stats, compute_noise_matrix, compute_snr_matrix, compute_delay_matrix
from .utils import CG
import numpy as np
import pickle

class EcogData:
    def __init__(self, path:str='data/'):
        data_package = np.load('data/preprocessed_data.npz', allow_pickle=True)
        self.stride = data_package['stride']
        self.filters = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'raw']
        self.adj_mat = data_package['adj_mat'] + np.eye(self.stride.shape[0]-1)*1.5
        self.weight = data_package['weight']
        self.fc = {}
        self.roi_mask = None

    def compute_roi_masking(self, ty_mask:str):
        if ty_mask == 'ch':
            self.roi_mask = ~np.eye(self.stride[-1], dtype=bool)
        elif ty_mask == 'cg':
            multiplicity = np.diff(self.stride).astype(int)
            self.roi_mask = ~np.diag(multiplicity == 1).astype(bool)
        else:
            raise AttributeError('Invalid mask type.')
        return self.roi_mask

    def get_sc_fc(self, ty:str):
        if not self.fc:
            raise RuntimeError('Missing FC data.')
        self.compute_roi_masking(ty)
        if ty == 'ch':
            sc = {band : self.weight[self.roi_mask] for band in self.filters}
            fc = {}
            for band in self.filters:
                if self.fc[band] is not None:
                    fc[band] = self.fc[band][self.roi_mask]
                else:
                    fc[band] = None
        elif ty == 'cg':
            sc = {band : self.adj_mat[self.roi_mask] for band in self.filters}
            fc = {}
            for band in self.filters:
                if self.fc[band] is not None:
                    fc[band] = CG(self.fc[band], self.stride)[self.roi_mask]
                else:
                    fc[band] = None
        else:
            raise AttributeError('Invalid mask type.')
        return sc, fc

class EcogTDMI(EcogData):
    def __init__(self, path:str='data/', dfname:str='tdmi_data_long.npz'):
        super().__init__(path)
        self.tdmi_mode = 'max'
        self.tdmi_data = np.load(path+dfname, allow_pickle=True)

    def init_data(self, snr_th_path:str=None, fname:str='snr_th.pkl'):
        self.compute_mi_stats()
        if snr_th_path is not None:
            with open(snr_th_path+fname, 'rb') as f:
                snr_th = pickle.load(f)
            for band in self.filters:
                snr_matrix = compute_snr_matrix(self.tdmi_data[band])
                noise_matrix = compute_noise_matrix(self.tdmi_data[band])
                snr_mask = snr_matrix >= snr_th[band]
                self.fc[band][~snr_mask] = noise_matrix[~snr_mask]
        return self

    def init_data_strict(self, snr_th_path:str=None)->None:
        self.compute_mi_stats()
        if snr_th_path is not None:
            with open(snr_th_path+'snr_th.pkl', 'rb') as f:
                snr_th = pickle.load(f)
            for band in self.filters:
                snr_matrix = compute_snr_matrix(self.tdmi_data[band])
                snr_mask = snr_matrix >= snr_th[band]
                self.fc[band][~snr_mask] = np.nan

    def compute_mi_stats(self):
        for band in self.filters:
            self.fc[band] = MI_stats(self.tdmi_data[band], self.tdmi_mode)

    def get_snr_mask(self, snr_th_path:str=None, fname:str='snr_th.pkl'):
        with open(snr_th_path+fname, 'rb') as f:
            snr_th = pickle.load(f)
        self.compute_roi_masking('ch')
        snr_mask = {}
        for band in self.filters:
            snr_matrix = compute_snr_matrix(self.tdmi_data[band])
            snr_mask[band] = snr_matrix >= snr_th[band]
            snr_mask[band] = snr_mask[band][self.roi_mask]
        return snr_mask

    def get_delay_matrix(self,):
        delay_matrix = {}
        for band in self.filters:
            delay_matrix[band] = compute_delay_matrix(self.tdmi_data[band])
        return delay_matrix

class EcogGC(EcogData):
    def __init__(self, path:str='data/'):
        super().__init__(path)
        self.gc_data = np.load(path+'gc_order_6.npz', allow_pickle=True)

    def init_data(self):
        for band in self.filters:
            if band in self.gc_data.files:
                self.fc[band] = self.gc_data[band].copy()
                self.fc[band][self.fc[band]<=0] = 1e-5
            else:
                self.fc[band] = None

class EcogCC(EcogData):
    def __init__(self, path:str='data/', dfname:str='cc.npz'):
        super().__init__(path)
        self.cc_data = np.load(path+dfname, allow_pickle=True)
        self.adj_mat = (self.adj_mat + self.adj_mat.T)/2
        self.weight = (self.weight + self.weight.T)/2

    def init_data(self):
        for band in self.filters:
            self.fc[band] = np.abs(self.cc_data[band])

class EcogTDCC(EcogTDMI):
    def __init__(self, path:str='data/', dfname:str='tdcc.npz'):
        super().__init__(path, dfname)
        data_buffer = {}
        for band in self.filters:
            data_buffer[band] = np.abs(self.tdmi_data[band])
        self.tdmi_data = data_buffer.copy()