import argparse
from lib.matrics import get_adjacency_matrix, get_distance_adjacency_matrix
from lib.utils import eval_res, get_eigenvector


argparse = argparse.ArgumentParser()
argparse.add_argument('--num_of_vertices', type=int, default=307)
argparse.add_argument('--distance_filename', type=str, default='data/distance.csv')
argparse.add_argument('--adjacency_type', type=str, default='distance')
argparse.add_argument('--laplacian_type', type=str, default='scale')
argparse.add_argument('--num_of_features', type=int, default=10)
argparse.add_argument('--num_of_cluster', type=int, default=5)
args = argparse.parse_args()

num_of_vertices, num_of_features, num_of_cluster = args.num_of_vertices,args.num_of_features, args.num_of_cluster
distance_filename = args.distance_filename
adjacency_type, laplacian_type = args.adjacency_type, args.laplacian_type

if adjacency_type == 'distance':
    adj_mx = get_distance_adjacency_matrix(distance_filename, num_of_vertices)
else:
    adj_mx = get_adjacency_matrix(distance_filename, num_of_vertices)

eigenvector = get_eigenvector(adj_mx, num_of_features, laplacian_type)

if __name__ == '__main__':
    result = eval_res(eigenvector, num_of_cluster)
    print("the number of features in eigenvector is:", num_of_features)
    print("the number of cluster is:", num_of_cluster)
    print("the Calinski Harabasz score is:", result)