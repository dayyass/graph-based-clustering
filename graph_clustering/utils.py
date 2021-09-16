import numpy as np
from check import _check_matrix
from sklearn.metrics import pairwise_distances


def _pairwise_distances(
    X: np.ndarray,
    metric: str = "euclidean",  # TODO: extend callable
    n_jobs: int = -1,
) -> np.ndarray:
    """Compute the pairwise distance matrix from a matrix X.

    Args:
        X (np.ndarray): A matrix.
        metric (str, optional): The metric to use when calculating distance between instances in a feature array. Defaults to "euclidean".
        n_jobs (int, optional): The number of jobs to use for the computation. Defaults to -1.

    Returns:
        np.ndarray: The pairwise distance matrix.
    """

    assert _check_matrix(X)

    distances = pairwise_distances(X=X, metric=metric, n_jobs=n_jobs)

    return distances
