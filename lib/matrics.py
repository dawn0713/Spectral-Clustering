import numpy as np
import pandas as pd
import csv


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    """
    create adjacency matrix using distance.csv file
    :param distance_df_filename: the dir_path of distance.csv
    :param num_of_vertices: number of the nodes in graph
    :return: np.ndarray, 0-1 adjacency matrix,adj_mx[i][j] == 1 if i is the neighbor of j else 0
    """

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        edges = [(int(i[0]), int(i[1])) for i in reader]

    adj_mx = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    for i, j in edges:
        adj_mx[i, j] = 1

    return adj_mx


def get_distance_adjacency_matrix(distance_df_filename, num_of_vertices):
    """
    create adjacency matrix using Gaussian Kernel
    :param distance_df_filename: str, the dir_path of distance.csv
    :param num_of_vertices: int, number of the nodes in graph
    :return: np.ndarray, adjacency matrix
    """

    distance_df = pd.read_csv(distance_df_filename, dtype={'from': 'int', 'to': 'int'})
    dist_mx = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)

    # init the dist_mx with inf
    dist_mx[:] = np.inf

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        dist_mx[int(row[0]), int(row[1])] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    return adj_mx


def cal_laplacian_matrix(adjacent_matrix):
    """
    calculate the Laplacian matrix of the adjacent matrix
    :param adjacent_matrix: np.ndarray
    :return: np.ndarray, L = D - A
    """
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacent_matrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacent_matrix

    return laplacianMatrix


def cal_scale_laplacian_matrix(adjacent_matrix):
    np.seterr(divide='ignore', invalid='ignore')
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacent_matrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacent_matrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)