#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_SNR_PATH="tdmi_snr_analysis/"
mkdir -p $TDMI_SNR_PATH

# determine SNR threshold
python tdmi_snr_distribution.py
python tdmi_snr_and_trace.py

# Channel-wise Analysis
python tdmi_snr_causal.py $TDMI_SNR_PATH
python tdmi_snr_auc_threshold.py $TDMI_SNR_PATH
python tdmi_snr_mi_s.py $TDMI_SNR_PATH

# get pval.npz, gap_th.pkl, roc.pkl
python get_w_threshold.py
python tdmi_rank.py

# reconstruction
python WA_v3.py
python WA_v3_ppv.py
python WA_v3_roc.py
python WA_v3_summary.py
python WA_v5_TPTN_feature.py

# Coarse-Grain Analysis
python tdmi_snr_causal_cg.py $TDMI_SNR_PATH
python tdmi_snr_auc_threshold_cg.py $TDMI_SNR_PATH
python tdmi_snr_mi_s_cg.py $TDMI_SNR_PATH

# get pval.npz, gap_th.pkl, roc.pkl
python get_w_threshold_cg.py
python tdmi_rank_cg.py

# reconstruction
python WA_v3_cg.py
python WA_v3_ppv_cg.py
python WA_v3_roc_cg.py
python WA_v3_summary_cg.py

# plot geometric graph
python plot_geometric_graph.py

echo "[INFO]: processing done!"