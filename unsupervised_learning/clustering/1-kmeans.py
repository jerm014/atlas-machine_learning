#!/usr/bin/env python3
"""documenatation"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Perform K-means clustering on a dataset.

    Args:
       X: numpy.ndarray of shape (n,d) containing dataset
       k: int, positive integer containing number of clusters
       iterations: int, maximum number of iterations to perform

    Returns:
       tuple(numpy.ndarray, numpy.ndarray) containing centroids and labels
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    labels, new_labels, centroids = None, None, None

    try:
        for _ in range(iterations):
            #if cluster contains no data points during the update reinit the centroids:
            cluster_sizes = np.bincount(new_labels, minlength=k)

            # Or more simply:
            has_empty = (cluster_sizes == 0).any()
            if centroids is None or has_empty is not None:
                centroids = initialize(X, k)


            distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(2))
            new_labels = np.argmin(distances, axis=1)
            old_centroids = centroids.copy()

            for j in range(k):
                points = X[new_labels == j]
                centroids[j] = (
                    np.random.uniform(low=mins, high=maxs, size=d)
                    if points.shape[0] == 0
                    else np.mean(points, axis=0)
                )

            if np.all(old_centroids == centroids):
                labels = new_labels
                break

            labels = new_labels

        return centroids, labels

    except Exception:
        return None, None


def initialize(X, k):
    """Initialize K-means cluster centroids using uniform distribution.

    Args:
        X: numpy.ndarray of shape (n,d) containing dataset for clustering
        k: int, positive integer containing number of clusters

    Returns:
        numpy.ndarray of shape (k,d) with initialized centroids, None on
        failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        # Find min values along each dimension
        # (like finding leftmost and bottommost points)
        mins = X.min(axis=0)
        # Find max values along each dimension
        # (like finding rightmost and topmost points)
        maxs = X.max(axis=0)
        # Create k random points, each having d dimensions
        # For each dimension, values will be between that dimension's
        # min & max
        centroids = np.random.uniform(
                                      low=mins,
                                      high=maxs,
                                      size=(k, X.shape[1]))
        return centroids
    except Exception:
        return None
