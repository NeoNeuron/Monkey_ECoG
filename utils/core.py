# Author: Kai Chen
from argparse import ArgumentTypeError
from .tdmi import MI_stats, compute_noise_matrix, compute_snr_matrix
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
            raise ArgumentTypeError('Invalid mask type.')

    def get_sc_fc(self, ty:str):
        if not self.fc:
            raise RuntimeError('Missing FC data.')
        self.compute_roi_masking(ty)
        if ty == 'ch':
            sc = {band : self.weight[self.roi_mask] for band in self.filters}
            fc = {band : self.fc[band][self.roi_mask] for band in self.filters}
        elif ty == 'cg':
            sc = {band : self.adj_mat[self.roi_mask] for band in self.filters}
            fc = {band : CG(self.fc[band], self.stride)[self.roi_mask] for band in self.filters}
        else:
            raise ArgumentTypeError('Invalid mask type.')
        return sc, fc

class EcogTDMI(EcogData):
    def __init__(self, path:str='data/', dfname:str='tdmi_data_long.npz'):
        super().__init__(path)
        self.tdmi_mode = 'max'
        self.tdmi_data = np.load(path+dfname, allow_pickle=True)

    def init_data(self, snr_th_path:str=None)->None:
        self.compute_mi_stats()
        if snr_th_path is not None:
            with open(snr_th_path+'snr_th.pkl', 'rb') as f:
                snr_th = pickle.load(f)
            self.compute_snr_masking(snr_th)

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

    def compute_snr_masking(self, snr_th):
        for band in self.filters:
            snr_matrix = compute_snr_matrix(self.tdmi_data[band])
            noise_matrix = compute_noise_matrix(self.tdmi_data[band])
            snr_mask = snr_matrix >= snr_th[band]
            self.fc[band][~snr_mask] = noise_matrix[~snr_mask]

    def get_snr_mask(self, snr_th_path:str):
        with open(snr_th_path+'snr_th.pkl', 'rb') as f:
            snr_th = pickle.load(f)
        self.compute_roi_masking('ch')
        snr_mask = {}
        for band in self.filters:
            snr_matrix = compute_snr_matrix(self.tdmi_data[band])
            snr_mask[band] = snr_matrix >= snr_th[band]
            snr_mask[band] = snr_mask[band][self.roi_mask]
        return snr_mask

class EcogGC(EcogData):
    def __init__(self, path:str='data/'):
        super().__init__(path)
        self.gc_data = np.load(path+'gc_order_6.npz', allow_pickle=True)

    def init_data(self):
        for band in self.filters:
            self.fc[band] = self.gc_data[band].copy()
            self.fc[band][self.fc[band]<=0] = 1e-5

class EcogCC(EcogData):
    def __init__(self, path:str='data/', dfname:str='cc.npz'):
        super().__init__(path)
        self.cc_data = np.load(path+dfname, allow_pickle=True)
        self.adj_mat = (self.adj_mat + self.adj_mat.T)/2
        self.weight = (self.weight + self.weight.T)/2

    def init_data(self):
        for band in self.filters:
            self.fc[band] = np.abs(self.cc_data[band])