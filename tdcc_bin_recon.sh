#!/bin/zsh
echo "[INFO]: processing started!"
TDCC_PATH="tdmi_snr_analysis/"
mkdir -p $TDCC_PATH

python ch_fc_rank_tdcc.py
python cg_fc_rank_tdcc.py

# get pval.npz, gap_th.pkl, roc.pkl
python ch_get_thresholds_tdcc.py
python cg_get_thresholds_tdcc.py

# Channel-wise Analysis
python ch_overview_tdcc.py $TDCC_PATH
python ch_auc_th_tdcc.py $TDCC_PATH
python ch_fit_tdcc.py $TDCC_PATH

# Coarse-Grain Analysis
python cg_overview_tdcc.py $TDCC_PATH
python cg_auc_th_tdcc.py $TDCC_PATH
python cg_fit_tdcc.py $TDCC_PATH

# reconstruction
python ch_bin_recon_tdcc.py
python ch_bin_recon_2_md_tdcc.py
python ch_bin_recon_ppv_tdcc.py
python ch_bin_recon_figure_tdcc.py

python cg_bin_recon_tdcc.py
python cg_bin_recon_2_md_tdcc.py
python cg_bin_recon_ppv_tdcc.py
python cg_bin_recon_figure_tdcc.py

echo "[INFO]: processing done!"