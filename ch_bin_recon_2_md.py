# Author: Kai Chen
# Description: Convert binary reconstruction results to markdown format.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - FC recon matrix masked by 3 types of FC thresholding mask;
# *   - All normalized to 0-1 valued matrix;
from utils.roc import ROC_matrix
import numpy as np

def pkl2md(fname:str, sc_mask:list, fc_mask:dict):
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    if len(list(fc_mask.values())[0].shape) == 1:
        for key, item in fc_mask.items():
            fc_mask[key] = np.tile(item, (len(w_thresholds),1))
    with open(fname, 'w') as ofile:
        roc_data = np.zeros((w_thresholds.shape[0], len(fc_mask.keys()), 8,))
        for idx, sc in enumerate(sc_mask):
            print("## $w_{ij}>10^{%d}$ " % int(np.log10(w_thresholds[idx])), file=ofile)
            print(f'p = {np.sum(sc)/sc.shape[0]:6.3f}', file=ofile)
            print('| band | TP | FP | FN | TN | Corr | TPR | FPR | PPV |', file=ofile)
            print('|------|----|----|----|----|------| --- | --- | --- |', file=ofile)

            union_mask = np.zeros_like(sc, dtype=bool)
            for iidx, band in enumerate(fc_mask.keys()):
                if band != 'raw':
                    union_mask += fc_mask[band][idx]
                TP, FP, FN, TN = ROC_matrix(sc, fc_mask[band][idx])
                CORR = np.corrcoef(sc, fc_mask[band][idx])[0, 1]
                if np.isnan(CORR):
                    CORR = 0.
                roc_data[idx, iidx, :] = [TP,FP,FN,TN,CORR,TP/(TP+FN),FP/(FP+TN),TP/(TP+FP)]
                print('|%s|%d|%d|%d|%d|%6.3f|%6.3f|%6.3f|%6.3f|' % (band, *roc_data[idx, iidx, :]), file=ofile)
            print(f'**CORR = {np.corrcoef(sc, union_mask)[0, 1]:6.3f}**', file=ofile)

    np.save(fname.replace('md', 'npy'), roc_data)

if __name__ == '__main__':
    import pickle
    path = 'tdmi_snr_analysis/'
    # Load SC and FC data
    # ==================================================
    fnames = ['recon_fit_tdmi.pkl', 'recon_roc_tdmi.pkl', 'recon_gap_tdmi.pkl']
    # ==================================================

    for fname in fnames:
        with open(path+fname, 'rb') as f:
            sc_mask = pickle.load(f)
            fc_mask = pickle.load(f)
            pkl2md(path + fname.replace('pkl', 'md'), sc_mask, fc_mask)