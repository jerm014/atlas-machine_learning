#!/usr/bin/env python3
"""
Implementation of K-means clustering algorithm.
But we get to use sklearn.
"""

import sklearn.cluster


def kmeans(X, k):
    """Perform K-means clustering on input data.

    Args:
       X: numpy.ndarray of shape (n, d) containing the dataset
       k: int, number of clusters

    Returns:
       tuple containing:
           - numpy.ndarray of shape (k, d) with cluster centroids
           - numpy.ndarray of shape (n,) with cluster assignments
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
