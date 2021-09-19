from typing import Dict, Tuple

import numpy as np
from sklearn import datasets


def prepare_sklearn_clustering_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

    # ============
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
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
    no_structure = (
        np.random.rand(n_samples, 2),
        np.zeros(n_samples, dtype=int),
    )

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

    sklearn_clustering_datasets = {}

    sklearn_clustering_datasets["noisy_circles"] = noisy_circles
    sklearn_clustering_datasets["noisy_moons"] = noisy_moons
    sklearn_clustering_datasets["blobs"] = blobs
    sklearn_clustering_datasets["no_structure"] = no_structure
    sklearn_clustering_datasets["aniso"] = aniso
    sklearn_clustering_datasets["varied"] = varied

    return sklearn_clustering_datasets
