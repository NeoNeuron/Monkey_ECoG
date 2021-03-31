#!/bin/zsh
echo "[INFO]: processing started!"
GC_PATH="data_preprocessing_46_region/"
mkdir -p $GC_PATH

# preprocess data
python gc_scan.py $GC_PATH 6 False
python gc_scan.py $GC_PATH 6 True

# Channel-wise Analysis
python ch_overview_gc.py $GC_PATH 6
python ch_auc_th_gc.py $GC_PATH 6
python ch_fit_gc.py $GC_PATH 6

# interarea
python ch_overview_gc.py $GC_PATH 6 True
python ch_auc_th_gc.py $GC_PATH 6 True
python ch_fit_gc.py $GC_PATH 6 True

# Coarse-Grain Analysis
python cg_overview_gc.py $GC_PATH 6
python cg_aud_th_gc.py $GC_PATH 6
python cg_fit_gc.py $GC_PATH 6

echo "[INFO]: processing done!"