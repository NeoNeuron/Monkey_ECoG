import numpy as np 
from scipy.io import loadmat
from utils.utils import CG
import matplotlib.pyplot as plt 

path = 'data/'
stride = np.load(path+'preprocessed_data.npz', allow_pickle=True)['stride']

dat = loadmat(path + 'm_data_raw.mat')

fig = plt.figure(figsize=(12,5))
gs1 = fig.add_gridspec(nrows=1, ncols=1, 
	left=0.05, right=0.38, top=0.96, bottom=0.05, 
)
gs2 = fig.add_gridspec(nrows=1, ncols=2, 
	left=0.41, right=0.96, top=0.96, bottom=0.05, 
	wspace=0.15, hspace=0.20
)
ax = [fig.add_subplot(gs1[0]), fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1]) ]
pax = ax[0].pcolormesh(np.log10(dat['GC']+1e-3), cmap=plt.cm.Oranges)
plt.colorbar(pax, ax=ax[0], shrink=0.7)
ax[0].invert_yaxis()
ax[0].axis('scaled')
ax[0].set_title('Cond GC Value')

gc_thred = dat['gc_zero_line'][0,0]
print(gc_thred)
ax[1].pcolormesh((dat['GC']>=gc_thred), cmap=plt.cm.gray)
ax[1].invert_yaxis()
ax[1].axis('scaled')
ax[1].set_title('Binary Recon from Cond GC Value')

gc_cg = CG(dat['GC'], stride)
pax = ax[2].pcolormesh(gc_cg>gc_thred)
# plt.colorbar(pax, ax=ax[2])
ax[2].invert_yaxis()
ax[2].axis('scaled')
ax[2].set_title('CG Binary Recon from Cond GC Value')

# plt.tight_layout()
plt.savefig('test.png')