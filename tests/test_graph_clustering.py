import unittest

import numpy as np

from graph_clustering.check import check_adjacency_matrix, check_symmetric
from graph_clustering.utils import _pairwise_distances, distances_to_adjacency_matrix

X = np.array([[0, 1], [1, 0], [1, 1]])

distances = _pairwise_distances(X)

adjacency_matrix = distances_to_adjacency_matrix(
    distances=distances,
    threshold=1.25,
)


class TestCheck(unittest.TestCase):
    def test_check_symmetric(self):
        """Test check_symmetric"""

        self.assertFalse(check_symmetric(X))

        self.assertTrue(check_symmetric(distances))

        self.assertTrue(check_symmetric(adjacency_matrix))

    def test_check_adjacency_matrix(self):
        """Test check_adjacency_matrix"""

        self.assertFalse(check_adjacency_matrix(X))

        self.assertFalse(check_adjacency_matrix(distances))

        self.assertTrue(check_adjacency_matrix(adjacency_matrix))
