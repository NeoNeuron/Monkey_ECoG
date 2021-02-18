import numpy as np
import scipy.cluster.hierarchy as sch

def get_cluster_id(X):
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    return np.argsort(ind)

def get_sorted_mat(X, sort_id):
    X_sort = X.copy() 
    X_sort = X_sort[:, sort_id]
    X_sort = X_sort[sort_id, :]
    return X_sort


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    path = 'data_preprocessing_46_region/'
    data_package = np.load(path + 'preprocessed_data.npz', allow_pickle=True)
    adj_mat = data_package['adj_mat']
    adj_mat[adj_mat<5e-3] = 1e-6

    adj_mat_log = np.log10(adj_mat+1e-6)
    fig, ax = plt.subplots(1,2, figsize=(10,6))

    pax = ax[0].pcolormesh(adj_mat_log, cmap=plt.cm.rainbow)
    plt.colorbar(pax, ax=ax[0])
    ax[0].axis('scaled')
    ax[0].invert_yaxis()

    sort_id = get_cluster_id(adj_mat_log)
    adj_mat_sort = get_sorted_mat(adj_mat_log, sort_id)

    pax = ax[1].pcolormesh(adj_mat_sort, cmap=plt.cm.rainbow)
    plt.colorbar(pax, ax=ax[1])
    ax[1].axis('scaled')
    ax[1].invert_yaxis()

    plt.savefig('tmp/adj_mat.png')