[![tests](https://github.com/dayyass/graph-based-clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/dayyass/graph-based-clustering/actions/workflows/tests.yml)
[![linter](https://github.com/dayyass/graph-based-clustering/actions/workflows/linter.yml/badge.svg)](https://github.com/dayyass/graph-based-clustering/actions/workflows/linter.yml)
[![codecov](https://codecov.io/gh/dayyass/graph-based-clustering/branch/main/graph/badge.svg?token=ZVR4C5SRON)](https://codecov.io/gh/dayyass/graph-based-clustering)

[![python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://github.com/dayyass/graph-based-clustering#requirements)
[![release (latest by date)](https://img.shields.io/github/v/release/dayyass/graph-based-clustering)](https://github.com/dayyass/graph-based-clustering/releases/latest)
[![license](https://img.shields.io/github/license/dayyass/graph-based-clustering?color=blue)](https://github.com/dayyass/graph-based-clustering/blob/main/LICENSE)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-black)](https://github.com/dayyass/graph-based-clustering/blob/main/.pre-commit-config.yaml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pypi version](https://img.shields.io/pypi/v/graph-based-clustering)](https://pypi.org/project/graph-based-clustering)
[![pypi downloads](https://img.shields.io/pypi/dm/graph-based-clustering)](https://pypi.org/project/graph-based-clustering)

### Graph-Based Clustering

Graph-Based Clustering using connected components and minimum spanning trees.

Both clustering methods, supported by this library, are **transductive** - meaning they are not designed to be applied to new, unseen data.

### Installation

To install **graph-based-clustering** run:
```
pip install graph-based-clustering
```

### Usage

**graph-based-clustering** has two clustering methods:
- ConnectedComponentsClustering
- SpanTreeConnectedComponentsClustering

Both of these methods has sklearn-like `fit/fit_predict` interface.

#### ConnectedComponentsClustering

This method makes pairwise distances matrix on the input data, use *threshold* (parameter given by the user) to binarize pairwise distances matrix and make undirected graph, and then finds connected components to perform the clustering.

Required arguments:
- **threshold** - threshold to binarize pairwise distances matrix and make undirected graph

Optional arguments:
- **metric** - sklearn.metrics[pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) parameter (default: *"euclidean"*)
- **n_jobs** - sklearn.metrics[pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) parameter (default: *None*)

Example:

```python3
from graph_based_clustering import ConnectedComponentsClustering

X = np.array([[0, 1], [1, 0], [1, 1]])

clustering = ConnectedComponentsClustering(
    threshold=0.275,
    metric="euclidean",
    n_jobs=-1,
)

clustering.fit(X)
labels_pred = clustering.labels_

# alternative
labels_pred = clustering.fit_predict(X)
```

#### SpanTreeConnectedComponentsClustering

This method makes pairwise distances matrix on the input data, consider this matrix as a graph, finds minimum spanning trees, and finaly, to perform the clustering, makes graph with *n_clusters* (parameter given by the user) connected components by removing *n_clusters - 1* edges with highest weights.

Required arguments:
- **n_clusters** - the number of clusters to find

Optional arguments:
- **metric** - sklearn.metrics[pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) parameter (default: *"euclidean"*)
- **n_jobs** - sklearn.metrics[pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) parameter (default: *None*)

Example:

```python3
from graph_based_clustering import SpanTreeConnectedComponentsClustering

X = np.array([[0, 1], [1, 0], [1, 1]])

clustering = SpanTreeConnectedComponentsClustering(
    n_clusters=3,
    metric="euclidean",
    n_jobs=-1,
)

clustering.fit(X)
labels_pred = clustering.labels_

# alternative
labels_pred = clustering.fit_predict(X)
```

### Requirements
Python >= 3.7

### Citation
If you use **graph-based-clustering** in a scientific publication, we would appreciate references to the following BibTex entry:
```bibtex
@misc{dayyass2021graphbasedclustering,
    author       = {El-Ayyass, Dani},
    title        = {Graph-Based Clustering using connected components and spanning trees},
    howpublished = {\url{https://github.com/dayyass/graph-based-clustering}},
    year         = {2021}
}
```
