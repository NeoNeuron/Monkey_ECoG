#!/Users/kchen/miniconda3/bin/python
# Author: Kai Chen
# Institute: INS, SJTU
# Plot MI vs. connection strength.

import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=15
plt.rcParams['axes.labelsize'] = 15
from fcpy.core import EcogCC
from fcpy.utils import print_log
from fcpy.plot import gen_mi_s_figure
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
arg_default = {'path': 'data_preprocessing_46_region/',
                }
parser = ArgumentParser(prog='CG_mi_s',
                        description = "Generate figure for analysis of causality.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', default=arg_default['path'], nargs='?',
                    type = str, 
                    help = "path of working directory."
                    )
args = parser.parse_args()

start = time.time()
# Load SC and FC data
# ==================================================
data = EcogCC()
data.init_data()
sc, fc = data.get_sc_fc('cg')
# ==================================================

fig = gen_mi_s_figure(fc, sc, is_log=False)

for ax in fig.get_axes():
    handles, labels = ax.get_legend_handles_labels()
    labels = [item.replace('TDMI', 'CC') for item in labels]
    ax.legend(handles, labels)
[fig.get_axes()[i].set_ylabel(r'$log_{10}\left((CC)\right)$') for i in (0,4)]
[fig.get_axes()[i].set_xlabel('Weight') for i in (4,5,6)]
plt.tight_layout()

fname = f'cg_cc-s.png'
fig.savefig(args.path + fname)
print_log(f'Figure save to {args.path+fname:s}.', start)