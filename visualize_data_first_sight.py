import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
from scipy.io import loadmat

path = 'EcoG data-ChenYuHan/'
data_series = loadmat(path + 'r_c.mat')['r_c'][0]

num_region = data_series.shape[0]
multiplicity = np.zeros(num_region)
for i, item in enumerate(data_series):
    multiplicity[i] = item.shape[1]

w = loadmat(path + 'wei_r.mat')['wei_r']

adj_mat = np.zeros((num_region, num_region))
adj_mat[w[:,1].astype(int)-1, w[:,0].astype(int)-1] = w[:,2]

fig, ax = plt.subplots(1,1,figsize=(13,10))
pax = ax.pcolor(np.log10(adj_mat+1e-7))
ax.invert_yaxis()
plt.colorbar(pax)
plt.savefig(path + 'adj_mat.png')
plt.close()


fig, ax = plt.subplots(1,1, figsize=(4,3), dpi=100)
ax.hist(np.log10(w[:,2]), bins=50)
ax.set_xlabel(r'$\log_{10}(Weight)$')
ax.set_ylabel('Number of edges')
ax.xaxis.set_ticks([-5,-4, -3, -2, -1, 0])
plt.tight_layout()
plt.savefig(path + 'w_hist.png')
plt.close()
