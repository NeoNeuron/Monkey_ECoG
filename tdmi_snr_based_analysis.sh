#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_SNR_PATH="tdmi_snr_analysis/"
mkdir -p $TDMI_SNR_PATH

# Channel-wise Analysis
python tdmi_snr_causal.py $TDMI_SNR_PATH
python tdmi_snr_auc_threshold.py $TDMI_SNR_PATH
python tdmi_snr_mi_s.py $TDMI_SNR_PATH

# interarea analysis
python tdmi_snr_causal.py $TDMI_SNR_PATH max True
python tdmi_snr_auc_threshold.py $TDMI_SNR_PATH max True
python tdmi_snr_mi_s.py $TDMI_SNR_PATH max True

# Coarse-Grain Analysis
python tdmi_snr_causal_cg.py $TDMI_SNR_PATH
python tdmi_snr_auc_threshold_cg.py $TDMI_SNR_PATH
python tdmi_snr_mi_s_cg.py $TDMI_SNR_PATH

# delay analysis
python tdmi_delay_analysis.py $TDMI_SNR_PATH

echo "[INFO]: processing done!"