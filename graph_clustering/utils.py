import numpy as np
from check import _check_matrix, check_symmetric
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


def distances_to_adjacency_matrix(
    distances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert a pairwise distance matrix to adjacency_matrix given threshold.

    Args:
        distances (np.ndarray): A pairwise distance matrix.
        threshold (float): threshold to make grapg edges.

    Returns:
        np.ndarray: The adjacency_matrix.
    """

    assert check_symmetric(distances)

    N = distances.shape[0]

    adjacency_matrix = (distances < threshold).astype(int) - np.eye(N)

    return adjacency_matrix
