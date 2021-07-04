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
