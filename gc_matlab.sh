#!/bin/zsh
echo "[INFO]: processing started!"
GC_MATLAB_PATH="gc_matlab/"
mkdir -p $GC_MATLAB_PATH

# preprocess data
python mat2npy.py $GC_MATLAB_PATH

# Channel-wise Analysis

python gc_analysis.py $GC_MATLAB_PATH 6 True --filters beta gamma high_gamma
python gc_auc_threshold.py $GC_MATLAB_PATH 6 False
python gc_s.py $GC_MATLAB_PATH 6 False

# Coarse-Grain Analysis

python CG_gc_causal.py $GC_MATLAB_PATH 6 --filters beta gamma high_gamma
python CG_gc_auc_threshold.py $GC_MATLAB_PATH 6
python CG_gc_s.py $GC_MATLAB_PATH 6

echo "[INFO]: processing done!"