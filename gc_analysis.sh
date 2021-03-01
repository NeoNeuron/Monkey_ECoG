#!/bin/zsh
echo "[INFO]: processing started!"
GC_PATH="data_preprocessing_46_region/"
mkdir -p $GC_PATH

# preprocess data
python preprocessing_v2.py $GC_PATH
python gc_scan.py $GC_PATH 6 False
python gc_scan.py $GC_PATH 6 True

# Channel-wise Analysis
python gc_analysis.py $GC_PATH 6
python gc_analysis.py $GC_PATH 6 True
python gc_auc_threshold.py $GC_PATH 6
python gc_auc_threshold.py $GC_PATH 6 True
python gc_s.py $GC_PATH 6
python gc_s.py $GC_PATH 6 True

# Coarse-Grain Analysis
python CG_gc_causal.py $GC_PATH 6
python CG_gc_auc_threshold.py $GC_PATH 6
python CG_gc_s.py $GC_PATH 6

echo "[INFO]: processing done!"