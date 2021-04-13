#!/bin/zsh
echo "[INFO]: processing started!"
CC_PATH="tdmi_snr_analysis/"
mkdir -p $CC_PATH

python ch_fc_rank_cc.py
python cg_fc_rank_cc.py

# get pval.npz, gap_th.pkl, roc.pkl
python ch_get_thresholds_cc.py
python cg_get_thresholds_cc.py

# Channel-wise Analysis
python ch_overview_cc.py $CC_PATH
python ch_auc_th_cc.py $CC_PATH
python ch_fit_cc.py $CC_PATH

# Coarse-Grain Analysis
python cg_overview_cc.py $CC_PATH
python cg_auc_th_cc.py $CC_PATH
python cg_fit_cc.py $CC_PATH

# reconstruction
python ch_bin_recon_cc.py
python ch_bin_recon_2_md_cc.py
python ch_bin_recon_ppv_cc.py
python ch_bin_recon_figure_cc.py

python cg_bin_recon_cc.py
python cg_bin_recon_2_md_cc.py
python cg_bin_recon_ppv_cc.py
python cg_bin_recon_figure_cc.py

echo "[INFO]: processing done!"