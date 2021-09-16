import numpy as np

from graph_clustering.utils import _pairwise_distances, distances_to_adjacency_matrix

X = np.array([[0, 1], [1, 0], [1, 1]])

distances = _pairwise_distances(X)

adjacency_matrix = distances_to_adjacency_matrix(
    distances=distances,
    threshold=1.25,
)
