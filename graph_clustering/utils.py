import numpy as np


def _check_matrix(a: np.ndarray) -> bool:
    """Check if np.ndarray is a matrix.

    Args:
        a (np.ndarray): np.ndarray to check.

    Returns:
        bool: np.ndarray is a matrix.
    """

    return a.ndim == 2


def _check_matrix_is_square(a: np.ndarray) -> bool:
    """Check if a matrix is square.

    Args:
        a (np.ndarray): A matrix to check.

    Returns:
        bool: A matrix is square.
    """

    M, N = a.shape

    return M == N


def _check_square_matrix_is_symmetric(
    a: np.ndarray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """Check if a square matrix is symmetric.

    Args:
        a (np.ndarray): A square matrix to check.
        rtol (float, optional): The relative tolerance parameter. Defaults to 1e-05.
        atol (float, optional): The absolute tolerance parameter. Defaults to 1e-08.

    Returns:
        bool: A square matrix is symmetric.
    """

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_symmetric(a: np.ndarray) -> bool:
    """Check if a matrix is symmetric.

    Args:
        a (np.ndarray): A matrix to check.

    Returns:
        bool: A matrix is symmetric.
    """

    is_matrix = _check_matrix(a)
    is_square = _check_matrix_is_square(a)
    is_symmetric = _check_square_matrix_is_symmetric(a)

    return is_matrix and is_square and is_symmetric
