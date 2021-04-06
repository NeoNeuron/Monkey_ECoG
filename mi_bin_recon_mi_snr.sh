#!/bin/zsh
echo "[INFO]: processing started!"
TDMI_SNR_PATH="tdmi_snr_analysis/"
mkdir -p $TDMI_SNR_PATH

# determine SNR threshold
# python tdmi_snr_distribution.py
python tdmi_snr_and_trace.py

# other analysis
python ch_fc_rank_mi.py
python cg_fc_rank_mi.py

# get pval.npz, gap_th.pkl, roc.pkl
python ch_get_thresholds.py
python cg_get_thresholds.py

# reconstruction
python ch_bin_recon.py
python ch_bin_recon_2_md.py
python ch_bin_recon_ppv.py
python ch_bin_recon_figure.py

# reconstruction
python cg_bin_recon.py
python cg_bin_recon_2_md.py
python cg_bin_recon_ppv.py
python cg_bin_recon_figure.py


# plot geometric graph
# python plot_geometric_graph.py

echo "[INFO]: processing done!"