import unittest

import numpy as np
from parameterized import parameterized

from graph_clustering.check import check_adjacency_matrix, check_symmetric
from graph_clustering.main import ConnectedComponentsClustering
from graph_clustering.utils import _pairwise_distances, distances_to_adjacency_matrix


class TestCheck(unittest.TestCase):

    X = np.array([[0, 1], [1, 0], [1, 1]])

    distances = _pairwise_distances(X)

    adjacency_matrix = distances_to_adjacency_matrix(
        distances=distances,
        threshold=1.25,
    )

    def test_check_symmetric(self):
        self.assertFalse(check_symmetric(self.X))
        self.assertTrue(check_symmetric(self.distances))
        self.assertTrue(check_symmetric(self.adjacency_matrix))

    def test_check_adjacency_matrix(self):
        self.assertFalse(check_adjacency_matrix(self.X))
        self.assertFalse(check_adjacency_matrix(self.distances))
        self.assertTrue(check_adjacency_matrix(self.adjacency_matrix))


class TestConnectedComponentsClustering(unittest.TestCase):
    @parameterized.expand(
        [
            (1.5, 1, [0, 0, 0]),
            (1.25, 1, [0, 0, 0]),
            (1.0, 3, [0, 1, 2]),
            (0.75, 3, [0, 1, 2]),
        ]
    )
    def test_mini_dataset(self, threshold, components_, labels_):

        X = np.array([[0, 1], [1, 0], [1, 1]])

        clustering = ConnectedComponentsClustering(
            threshold=threshold,
            metric="euclidean",
            n_jobs=-1,
        )

        clustering.fit(X)

        self.assertEqual(clustering.components_, components_)
        self.assertTrue(np.allclose(clustering.labels_, labels_))

        labels = clustering.fit_predict(X)

        self.assertTrue(np.allclose(clustering.labels_, labels))
