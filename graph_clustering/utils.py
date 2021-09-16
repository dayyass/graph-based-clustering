import numpy as np


def check_matrix(a: np.ndarray) -> bool:
    """Check if np.ndarray is a matrix.

    Args:
        a (np.ndarray): np.ndarray to check.

    Returns:
        bool: np.ndarray is a matrix.
    """

    return a.ndim == 2


def check_square(a: np.ndarray) -> bool:
    """Check if a matrix is square.

    Args:
        a (np.ndarray): A matrix to check.

    Returns:
        bool: A matrix is square.
    """

    M, N = a.shape

    return M == N


def check_symmetric(
    a: np.ndarray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """Check if a matrix is symmetric.

    Args:
        a (np.ndarray): A matrix to check.
        rtol (float, optional): The relative tolerance parameter. Defaults to 1e-05.
        atol (float, optional): The absolute tolerance parameter. Defaults to 1e-08.

    Returns:
        bool: A matrix is symmetric.
    """

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_distances(distances: np.ndarray) -> bool:
    """Check if a distances matrix is correct.

    Args:
        distances (np.ndarray): A distances matrix.

    Returns:
        bool: A distances matrix is correct.
    """

    is_matrix = check_matrix(distances)
    is_square = check_square(distances)
    is_symmetric = check_symmetric(distances)

    return is_matrix and is_square and is_symmetric
