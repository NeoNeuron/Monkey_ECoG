#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_PATH="data_preprocessing_46_region/"
mkdir -p $TDMI_PATH

# Channel-wise Analysis
python ch_overview_mi.py $TDMI_PATH
python ch_auc_th_mi.py $TDMI_PATH
python ch_fit_mi.py $TDMI_PATH

# Coarse-Grain Analysis
python cg_overview_mi.py $TDMI_PATH
python cg_auc_th_mi.py $TDMI_PATH
python cg_fit_mi.py $TDMI_PATH

# delay analysis
python tdmi_delay_analysis.py $TDMI_PATH

echo "[INFO]: processing done!"