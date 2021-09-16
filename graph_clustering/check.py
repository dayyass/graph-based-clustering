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


def _check_square_matrix_is_symmetric(a: np.ndarray) -> bool:
    """Check if a square matrix is symmetric.

    Args:
        a (np.ndarray): A square matrix to check.
        rtol (float, optional): The relative tolerance parameter. Defaults to 1e-05.
        atol (float, optional): The absolute tolerance parameter. Defaults to 1e-08.

    Returns:
        bool: A square matrix is symmetric.
    """

    return np.allclose(a, a.T)


def check_symmetric(a: np.ndarray) -> bool:
    """Check if a matrix is symmetric.

    Args:
        a (np.ndarray): A matrix to check.

    Returns:
        bool: A matrix is symmetric.
    """

    is_matrix = _check_matrix(a)
    is_matrix_square = _check_matrix_is_square(a)
    is_square_matrix_symmetric = _check_square_matrix_is_symmetric(a)

    return np.all([is_matrix, is_matrix_square, is_square_matrix_symmetric])


def _check_binary(a: np.ndarray) -> bool:
    """Check if np.ndarray is binary.

    Args:
        a (np.ndarray): np.ndarray to check.

    Returns:
        bool: np.ndarray is binary.
    """

    return ((a == 0) | (a == 1)).all()


def check_adjacency_matrix(a: np.ndarray) -> bool:
    """Check if a matrix is adjacency_matrix.

    Args:
        a (np.ndarray): A matrix to check.

    Returns:
        bool: A matrix is adjacency_matrix.
    """

    is_symmetric = check_symmetric(a)

    is_binary = _check_binary(a)
    is_zero_diag = not np.any(np.diag(a))

    return np.all([is_symmetric, is_binary, is_zero_diag])
