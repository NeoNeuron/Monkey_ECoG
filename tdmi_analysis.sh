#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_PATH="data_preprocessing_46_region/"
mkdir -p $TDMI_PATH

# Channel-wise Analysis
python draw_causal_distribution_v2.py $TDMI_PATH
python plot_auc_threshold.py $TDMI_PATH
python plot_mi_s.py $TDMI_PATH

# Coarse-Grain Analysis
python CG_causal_distribution.py $TDMI_PATH
python CG_auc_threshold.py $TDMI_PATH
python CG_mi_s.py $TDMI_PATH

# delay analysis
python tdmi_delay_analysis.py $TDMI_PATH

echo "[INFO]: processing done!"