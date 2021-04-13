#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_SNR_PATH="tdmi_snr_analysis/"
mkdir -p $TDMI_SNR_PATH

# other analysis
python ch_fc_rank_mi.py
python ch_fc_rank_gc.py
python ch_fc_rank_cc.py
python cg_fc_rank_mi.py
python cg_fc_rank_gc.py
python cg_fc_rank_cc.py

echo "[INFO]: processing done!"