import unittest

import numpy as np
from parameterized import parameterized
from sklearn import datasets

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
    @classmethod
    def setUpClass(cls):

        """https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html"""

        # ============
        # Generate datasets. We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
        # ============

        np.random.seed(0)

        n_samples = 1500
        noisy_circles = datasets.make_circles(
            n_samples=n_samples,
            factor=0.5,
            noise=0.05,
        )
        noisy_moons = datasets.make_moons(
            n_samples=n_samples,
            noise=0.05,
        )
        blobs = datasets.make_blobs(
            n_samples=n_samples,
            random_state=8,
        )
        no_structure = np.random.rand(n_samples, 2), None

        # Anisotropicly distributed data
        random_state = 170
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            random_state=random_state,
        )
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=[1.0, 2.5, 0.5],
            random_state=random_state,
        )

        cls.noisy_circles = noisy_circles
        cls.noisy_moons = noisy_moons
        cls.blobs = blobs
        cls.no_structure = no_structure
        cls.aniso = aniso
        cls.varied = varied

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
