# Spectral-Clustering
A python implementation of Spectral Clustering, using PEMS04 dataset

### Usage
 ```shell
  python main.py
  ```


### Steps of Spectral Clustering
Step1: Construct adjacency matrix by distance.csv

There are two kinds of adjacency matrices, one is **0-1 adjacency matrices** which indicating the neighbor relationship between nodes, 
and the other is **distance adjacency matrix** created based on the Gaussian kernel function.

Step2: Calculate the Laplacian matrix of the adjacency matrix

Step3: Calculate the eigenvector of the Laplacian matrix

Step4: Clustering using kmeans

Step5: Calculate the calinski harabasz score
