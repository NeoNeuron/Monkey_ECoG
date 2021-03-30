#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_SNR_PATH="tdmi_snr_analysis/"
mkdir -p $TDMI_SNR_PATH

# determine SNR threshold
python tdmi_snr_distribution.py
python tdmi_snr_and_trace.py

# Channel-wise Analysis
python ch_overview_mi_snr.py $TDMI_PATH
python ch_auc_th_mi_snr.py $TDMI_PATH
python ch_fit_mi_snr.py $TDMI_PATH

# get pval.npz, gap_th.pkl, roc.pkl
python ch_get_thresholds.py

# reconstruction
python ch_bin_recon.py
python ch_bin_recon_2_md.py
python ch_bin_recon_ppv.py
python ch_bin_recon_figure.py

# Coarse-Grain Analysis
python cg_overview_mi_snr.py $TDMI_PATH
python cg_auc_th_mi_snr.py $TDMI_PATH
python cg_fit_mi_snr.py $TDMI_PATH

# get pval.npz, gap_th.pkl, roc.pkl
python cg_get_thresholds.py

# reconstruction
python cg_bin_recon.py
python cg_bin_recon_2_md.py
python cg_bin_recon_ppv.py
python cg_bin_recon_figure.py

# plot geometric graph
# python plot_geometric_graph.py

echo "[INFO]: processing done!"