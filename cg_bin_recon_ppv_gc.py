# Author: Kai Chen
# Description: Plot PPV figure for binary reconstruction results.
# * Key Notion:
# *   - weight matrix masked by weight threshold; (weight > threshold)
# *   - TDMI recon matrix masked by corresponding TDMI threshold;
# *   - All normalized to 0-1 valued matrix;

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['lines.linewidth'] = 0.5
    from fcpy.plot import plot_ppv_curves
    path = 'tdmi_snr_analysis/'
    fnames = ['recon_fit_gc_CG.npy', 'recon_gap_gc_CG.npy', 'recon_roc_gc_CG.npy']
    fnames = [path+fname for fname in fnames]
    plot_ppv_curves(fnames, path + f'cg_bin_recon_ppv_gc.png')