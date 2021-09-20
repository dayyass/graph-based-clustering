import unittest

import numpy as np
from parameterized import parameterized
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler

from graph_clustering.check import (
    _check_matrix,
    _check_matrix_is_square,
    _check_square_matrix_is_symmetric,
    check_adjacency_matrix,
    check_symmetric,
)
from graph_clustering.main import (
    ConnectedComponentsClustering,
    SpanTreeConnectedComponentsClustering,
)
from graph_clustering.utils import _pairwise_distances, distances_to_adjacency_matrix

from .utils import prepare_sklearn_clustering_datasets

np.random.seed(0)


class TestCheck(unittest.TestCase):

    # diag matrix - graph with loops
    diag_matrix = np.eye(3)

    X = np.array([[0, 1], [1, 0], [1, 1]])

    distances = _pairwise_distances(X)

    adjacency_matrix = distances_to_adjacency_matrix(
        distances=distances,
        threshold=1.25,
    )

    def test_basic_checks(self):
        # _check_matrix
        a = np.random.randn(1, 2, 3)
        self.assertFalse(_check_matrix(a))
        self.assertFalse(check_symmetric(a))

        # _check_matrix_is_square
        a = np.random.randn(2, 3)
        self.assertFalse(_check_matrix_is_square(a))
        self.assertFalse(check_symmetric(a))

        # _check_square_matrix_is_symmetric
        a = np.random.randn(2, 2)
        self.assertFalse(_check_square_matrix_is_symmetric(a))
        self.assertFalse(check_symmetric(a))

    def test_check_symmetric(self):
        # diag matrix
        self.assertTrue(check_symmetric(self.diag_matrix))

        # X
        self.assertFalse(check_symmetric(self.X))
        self.assertTrue(check_symmetric(self.distances))
        self.assertTrue(check_symmetric(self.adjacency_matrix))

    def test_check_adjacency_matrix(self):
        # diag matrix
        self.assertFalse(check_adjacency_matrix(self.diag_matrix))

        # X
        self.assertFalse(check_adjacency_matrix(self.X))
        self.assertFalse(check_adjacency_matrix(self.distances))
        self.assertTrue(check_adjacency_matrix(self.adjacency_matrix))


class TestConnectedComponentsClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sklearn_clustering_datasets = prepare_sklearn_clustering_datasets()
        cls.sklearn_clustering_datasets = sklearn_clustering_datasets

    @parameterized.expand(
        [
            (1.5, 1, [0, 0, 0]),
            (1.25, 1, [0, 0, 0]),
            (1.0, 3, [0, 1, 2]),
            (0.75, 3, [0, 1, 2]),
        ]
    )
    def test_mini_dataset(self, threshold, n_clusters, labels):

        X = np.array([[0, 1], [1, 0], [1, 1]])

        clustering = ConnectedComponentsClustering(
            threshold=threshold,
            metric="euclidean",
            n_jobs=-1,
        )

        clustering.fit(X)

        labels_pred = clustering.labels_
        n_clusters_pred = len(np.unique(labels_pred))

        self.assertEqual(n_clusters_pred, n_clusters)
        self.assertTrue(np.allclose(labels_pred, labels))

        labels_pred_2 = clustering.fit_predict(X)

        self.assertTrue(np.allclose(labels_pred_2, labels))

    def test_sklearn_clustering_datasets(self):

        for dataset_name, dataset in self.sklearn_clustering_datasets.items():

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            clustering = ConnectedComponentsClustering(
                threshold=0.275,
                metric="euclidean",
                n_jobs=-1,
            )

            labels_pred = clustering.fit_predict(X)

            score = rand_score(labels_true=y, labels_pred=labels_pred)

            if dataset_name in ["aniso", "varied"]:
                self.assertNotEqual(score, 1.0)
            else:
                self.assertEqual(score, 1.0)


class TestSpanTreeConnectedComponentsClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # sklearn_clustering_datasets
        sklearn_clustering_datasets = prepare_sklearn_clustering_datasets()
        cls.sklearn_clustering_datasets = sklearn_clustering_datasets

        # n_cluster for each dataset
        datasets_n_cluster = {}

        datasets_n_cluster["noisy_circles"] = 2
        datasets_n_cluster["noisy_moons"] = 2
        datasets_n_cluster["blobs"] = 3
        datasets_n_cluster["no_structure"] = 1
        datasets_n_cluster["aniso"] = 3
        datasets_n_cluster["varied"] = 3

        cls.datasets_n_cluster = datasets_n_cluster

    @parameterized.expand([(1,), (2,), (3,)])
    def test_mini_dataset(self, n_clusters):

        X = np.array([[0, 1], [1, 0], [1, 1]])

        clustering = SpanTreeConnectedComponentsClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            n_jobs=-1,
        )

        clustering.fit(X)

        labels_pred = clustering.labels_
        n_clusters_pred = len(np.unique(labels_pred))

        self.assertEqual(n_clusters_pred, n_clusters)

        labels_pred_2 = clustering.fit_predict(X)
        n_clusters_pred = len(np.unique(labels_pred_2))

        self.assertEqual(n_clusters_pred, n_clusters)

    def test_sklearn_clustering_datasets(self):

        for dataset_name, dataset in self.sklearn_clustering_datasets.items():

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            clustering = SpanTreeConnectedComponentsClustering(
                n_clusters=self.datasets_n_cluster[dataset_name],
                metric="euclidean",
                n_jobs=-1,
            )

            labels_pred = clustering.fit_predict(X)

            score = rand_score(labels_true=y, labels_pred=labels_pred)

            if dataset_name in ["aniso", "varied"]:
                self.assertNotEqual(score, 1.0)
            else:
                self.assertEqual(score, 1.0)
