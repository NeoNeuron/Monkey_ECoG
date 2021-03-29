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
python WA_v3.py
python WA_v3_ppv.py
python WA_v3_roc.py
python WA_v3_summary.py
python WA_v5_TPTN_feature.py

# Coarse-Grain Analysis
python cg_overview_mi_snr.py $TDMI_PATH
python cg_auc_th_mi_snr.py $TDMI_PATH
python cg_fit_mi_snr.py $TDMI_PATH

# get pval.npz, gap_th.pkl, roc.pkl
python cg_get_thresholds.py

# reconstruction
python WA_v3_cg.py
python WA_v3_ppv_cg.py
python WA_v3_roc_cg.py
python WA_v3_summary_cg.py

# plot geometric graph
python plot_geometric_graph.py

echo "[INFO]: processing done!"