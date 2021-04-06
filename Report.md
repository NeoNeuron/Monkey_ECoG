# Causal Results for Monkey awake resting state ECoG Data III

> *Kai Chen*

## Brief Sumamry on Previous Results

1. TDMI works way more better than Granger Causality(GC), especially for low band cases. Results across different band are consistent with each other.

    > The ROC performance (AUC value) (see in Figure below) for different banded data and binary classification thresholding shows that TDMI works better than GC.
    > ![fig1](tdmi_snr_analysis/snr_th-/auc-threshold_summary.png)
    > The results for coarse-grained analysis are similar.
    > ![fig2](tdmi_snr_analysis/snr_th-/auc-threshold_summary_cg.png)
    > Later, we will show that as for PPV(positive prediction values) TDMI consistently works better than GC.

1. The Signal-to-Noise Ratio(SNR) masking operations in TDMI preprocessing is quite effective and helpful for the binary reconstruction results, and is also applicable for practical usage of this reconstruction framework.

    > Without SNR masking operation, double peak distribution of TDMI inferred functional connectivity is absent. which rise difficulty in thresholding in reconstruction of binary adjacent matrix.
    > To do SNR masking, we need to determine an SNR threshold according the distribution of SNR values. Here, we give 2 types of SNR thresholding strategies: K-means and double Gaussian fitting. The SNR thresholding results are shown below (cyan ones are threshold of K-means, and purple ones are double Gaussian fitting's). Also, previous manually selected threshold values are plotted.
    ![fig3](./snr_dist_figure.png)
    Later, we will demonstrate that K-means works better considering the PPV performance.

1. TDMI inference gives high positive prediction value(PPV)s, meaning, strong inferred effective functional connectivity are dominant by strong anatomical connections(structure connectivity).

    > ![fig4](./ch_bin_recon_ppv_comp.png)
    > In the figure above, `p true` is the positive prediction rate for random guess (black lines). The PPV performance of TDMI inferences (red and orange lines) are significantly higher than `p true` value (10-20 % higher), while those of GC's are almost same as the performance of random guess. We also plot the PPV performance of conventional functional connectivity (defined by correlation coefficient).
    > Note that the performance of K-means generated TDMI inference results are consisitently better than those of double Gaussian fitting generated ones.
1. Binary reconstruction by bands does reveal some structures with potential corresponding functions.

   > We present some of the reconstructed binary matrix from inference of different banded data.
   > In the data file `w_binary_recon.mat`, each 2D array named by its corresponding bands (including `delta`, `theta`, `alpha`, `beta`, `gamma`, `high_gamma`, and `raw`, where `raw` for the unfiltered data) represents the reconstructed edges. Within each array, the 1st and 2nd array show the efferent and afferent channel indices of each reconstructed edges, ranging from `0` to `179`, i.e., the same order as the original anatomical dataset you offered.
   > We wonder whether the reconstructed effective functional connectivity for different band corresponds to specific functional sub-networks. Hopefully, it might consist with previously revealed relation between functional network and frequency band.
