#!/bin/zsh
echo "[INFO]: Start making preprocessed data!"
DATA_PATH="data/"
mkdir -p $DATA_PATH

# preprocess data
python preprocessing_v2.py $DATA_PATH
python tdmi_scan_v2.py $DATA_PATH
python SI_test.py $DATA_PATH

echo "[INFO]: Done!"