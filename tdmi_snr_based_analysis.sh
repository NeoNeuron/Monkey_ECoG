#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_SNR_PATH="tdmi_snr_analysis/"
mkdir -p $TDMI_SNR_PATH

# Channel-wise Analysis
python ch_overview_mi_snr.py $TDMI_PATH
python ch_auc_th_mi_snr.py $TDMI_PATH
python ch_fit_mi_snr.py $TDMI_PATH

# interarea analysis
python ch_overview_mi_snr.py $TDMI_PATH True
python ch_auc_th_mi_snr.py $TDMI_PATH True
python ch_fit_mi_snr.py $TDMI_PATH True

# Coarse-Grain Analysis
python cg_overview_mi_snr.py $TDMI_PATH
python cg_auc_th_mi_snr.py $TDMI_PATH
python cg_fit_mi_snr.py $TDMI_PATH

echo "[INFO]: processing done!"