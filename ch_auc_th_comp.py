#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot AUC vs. answer threshold.

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib as mpl 
    mpl.rcParams['font.size']=20
    mpl.rcParams['axes.labelsize']=25
    from fcpy.plot import gen_auc_threshold_figure
    from fcpy.utils import print_log
    import pickle
    path='tdmi_snr_analysis/'

    start = time.time()
    w_thresholds = np.logspace(-6, 0, num=7, base=10)
    fnames = [
        path + 'tdmi_kmean/ch_aucs.pkl', 
        path + 'tdmi_gauss/ch_aucs.pkl',
        path + 'gc_6/ch_aucs_gc_order_6.pkl', 
        path + 'cc/gc_aucs_order_6.pkl',
    ]
    colors = ['r', 'orange', 'royalblue', 'springgreen']
    labels = ['TDMI(KMeans)','TDMI(Gauss)', 'GC', 'CC' ]
    with open(fnames[0], 'rb') as f:
        aucs_tdmi_nomask = pickle.load(f)
        aucs_tdmi_kmean = pickle.load(f)
    with open(fnames[1], 'rb') as f:
        pickle.load(f)
        aucs_tdmi_gauss = pickle.load(f)
    with open(fnames[2], 'rb') as f:
        aucs_gc = pickle.load(f)
    with open(fnames[3], 'rb') as f:
        aucs_cc = pickle.load(f)
    
    fig = gen_auc_threshold_figure(aucs_tdmi_nomask, w_thresholds, colors='k', labels='TDMI(No Mask)')
    ax = np.array(fig.get_axes())
    aucs = [aucs_tdmi_kmean, aucs_tdmi_gauss, aucs_gc, aucs_cc]
    for auc, color, label in zip(aucs, colors, labels):
        gen_auc_threshold_figure(auc, w_thresholds, ax=ax, colors=color, labels=label)
    ax[0].set_ylim(0.45,0.85)
    # plot legend in the empty subplot
    handles, labels = ax[0].get_legend_handles_labels()
    ax[-1].set_visible(True)
    ax[-1].legend(handles, labels, loc=2, fontsize=20)
    ax[-1].axis('off')


    fname = f'ch_auc-threshold_all_summary.png'
    fig.savefig(path + fname)
    print_log(f'Figure save to {path+fname:s}.', start)