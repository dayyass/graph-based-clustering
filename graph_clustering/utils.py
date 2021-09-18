from typing import Callable, Optional, Union

import numpy as np
from sklearn.metrics import pairwise_distances

from .check import _check_matrix, check_symmetric


def _pairwise_distances(
    X: np.ndarray,
    metric: Union[str, Callable] = "euclidean",
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Compute the pairwise distance matrix from a matrix X.

    Args:
        X (np.ndarray): A matrix.
        metric (Union[str, Callable], optional): The metric to use when calculating distance between instances in a feature array.
            If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter,
            or a metric listed in sklearn pairwise.PAIRWISE_DISTANCE_FUNCTIONS. Defaults to "euclidean".
        n_jobs (Optional[int], optional): The number of jobs to use for the computation. Defaults to None.

    Returns:
        np.ndarray: The pairwise distance matrix.
    """

    assert _check_matrix(X)

    distances = pairwise_distances(X=X, metric=metric, n_jobs=n_jobs)

    return distances


def distances_to_adjacency_matrix(
    distances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert a pairwise distance matrix to adjacency_matrix given threshold.

    Args:
        distances (np.ndarray): A pairwise distance matrix.
        threshold (float): Threshold to make graph edges.

    Returns:
        np.ndarray: The adjacency_matrix.
    """

    assert check_symmetric(distances)

    N = distances.shape[0]

    adjacency_matrix = (distances < threshold).astype(int) - np.eye(N)

    return adjacency_matrix
