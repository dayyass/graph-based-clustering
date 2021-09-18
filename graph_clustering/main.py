from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .utils import _pairwise_distances, distances_to_adjacency_matrix

# from sklearn.base import BaseEstimator, ClusterMixin  # TODO


class GraphConnectedComponentsClustering:

    """Clustering with graph connected components."""

    def __init__(
        self,
        threshold: float,
        metric: Union[str, Callable] = "euclidean",
        n_jobs: Optional[int] = None,
    ) -> None:
        """Init graph clustering model.

        Args:
            threshold (float): Threshold to make graph edges.
            metric (Union[str, Callable], optional): The metric to use when calculating distance between instances in a feature array.
                If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter,
                or a metric listed in sklearn pairwise.PAIRWISE_DISTANCE_FUNCTIONS. Defaults to "euclidean".
            n_jobs (Optional[int], optional): The number of jobs to use for the computation. Defaults to None.
        """

        self.threshold = threshold
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(
        self,
        X: np.ndarray,
    ):
        """Fit graph clustering model.

        Args:
            X (np.ndarray): A matrix.
        """

        distances = _pairwise_distances(
            X=X,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

        adjacency_matrix = distances_to_adjacency_matrix(
            distances=distances,
            threshold=self.threshold,
        )

        graph = csr_matrix(adjacency_matrix)

        n_components, labels = connected_components(
            csgraph=graph,
            directed=True,
            return_labels=True,
        )

        self.components_ = n_components
        self.labels_ = labels

        return self

    def fit_predict(
        self,
        X: np.ndarray,
    ):
        """Fit graph clustering model and return labels.

        Args:
            X (np.ndarray): A matrix.
        """

        self.fit(X)
        return self.labels_
