#
# ! Author: Kai Chen
# * Notion:
# *     - Compute statistical quantities in ROC analysis and save them into csv files.

import numpy as np

path = 'tdmi_snr_analysis/'

title_set = [
    "## $w_{ij}=0$ ",
    "## $0 < w_{ij} \leq 10^{-4}$ ",
    "## $10^{-4} < w_{ij} \leq 10^{-2}$ ",
    "## $w_{ij} > 10^{-2}$ ",
    "## $w_{ij} > 0$ ",
]

# Calculate TP FP FN TN Corr TPR FPR FNR TNR PPV 
roc_data = np.load(path + 'weight_analysis_v3_cg.npy', allow_pickle=True)
roc_processed = np.zeros((5, 7, 6))
for i in range(5):
    roc_processed[i,:,0] = roc_data[i,:,0]/(roc_data[i,:,0] + roc_data[i,:,2])
    roc_processed[i,:,1] = roc_data[i,:,1]/(roc_data[i,:,1] + roc_data[i,:,3])
    roc_processed[i,:,2] = roc_data[i,:,2]/(roc_data[i,:,0] + roc_data[i,:,2])
    roc_processed[i,:,3] = roc_data[i,:,3]/(roc_data[i,:,1] + roc_data[i,:,3])
    roc_processed[i,:,4] = roc_data[i,:,0]/(roc_data[i,:,0] + roc_data[i,:,1])
    roc_processed[i,:,5] = (roc_data[i,:,0] + roc_data[i,:,3])/roc_data[i,:,:].sum(1)
    np.savetxt(path + f'roc_analysis_{i:d}.csv', np.hstack((roc_data[i], roc_processed[i])), fmt="%7d "*4+"%7.4f "+"%7.4f "*6)