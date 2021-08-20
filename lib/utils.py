import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from .matrics import cal_scale_laplacian_matrix, cal_laplacian_matrix


def get_eigenvector(adj_mx, num_of_features, type='scale'):
    if type == 'scale':
        Laplacian = cal_scale_laplacian_matrix(adj_mx)
    else:
        Laplacian = cal_laplacian_matrix()
    Laplacian = np.nan_to_num(Laplacian)
    lam, H = np.linalg.eig(Laplacian)
    H = H.real
    H = H[:, 0:num_of_features]
    return H


def sp_kmeans(H, cluster_nodes):
    """
    Cluster the nodes using Kmeans
    :param H: np.ndarry, Eigenvector of the Laplacian matrix
    :param cluster_nodes: int, cluster number
    :return: np.ndarry, label of each node
    """
    sp_kmeans = KMeans(n_clusters=cluster_nodes, n_init=30, random_state=9).fit(H)
    return sp_kmeans.labels_


def eval_res(H, cluster_num):
    idx = sp_kmeans(H, cluster_num)
    ch_score = metrics.calinski_harabasz_score(H, idx)
    return ch_score
