#!/bin/zsh
echo "[INFO]: processing started!"
GC_PATH="tdmi_snr_analysis/"
mkdir -p $GC_PATH

python ch_fc_rank_gc.py
python cg_fc_rank_gc.py

# get pval.npz, gap_th.pkl, roc.pkl
python ch_get_thresholds_gc.py
python cg_get_thresholds_gc.py

# Channel-wise Analysis
python ch_overview_gc.py $GC_PATH
python ch_auc_th_gc.py $GC_PATH
python ch_fit_gc.py $GC_PATH

# Coarse-Grain Analysis
python cg_overview_gc.py $GC_PATH
python cg_auc_th_gc.py $GC_PATH
python cg_fit_gc.py $GC_PATH

# reconstruction
python ch_bin_recon_gc.py
python ch_bin_recon_2_md_gc.py
python ch_bin_recon_ppv_gc.py
python ch_bin_recon_figure_gc.py

python cg_bin_recon_gc.py
python cg_bin_recon_2_md_gc.py
python cg_bin_recon_ppv_gc.py
python cg_bin_recon_figure_gc.py

echo "[INFO]: processing done!"