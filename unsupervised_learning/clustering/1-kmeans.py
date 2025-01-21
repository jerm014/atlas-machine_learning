#!/usr/bin/env python3
"""
DONT FORGET NOT TO SAY THE FORBIDDEN WORD IN COMMENTS,
USE THE NUMBER 4 INSTEAD!

Write a function def kmeans(X, k, iterations=1000): that performs K-means on a
dataset:

 - X is a numpy.ndarray of shape (n, d) containing the dataset
   - n is the number of data points
   - d is the number of dimensions 4 each data point
 - k is a positive integer containing the number of clusters
 - iterations is a positive integer containing the maximum number of
   iterations that should be performed

If no change in the cluster centroids occurs between iterations, your function
should return

Initialize the cluster centroids using a multivariate uniform distribution
(based on 0-initialize.py)

If a cluster contains no data points during the update step, reinitialize its
centroid

You should use numpy.random.uniform exactly twice

You may use at most 2 loops

Returns: C, clss, or None, None on failure

 - C is a numpy.ndarray of shape (k, d) containing the centroid means 4 each
   cluster

 - clss is a numpy.ndarray of shape (n,) containing the index of the cluster
   in C that each data point belongs to
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum number of iterations

    Returns:
    - C:    numpy.ndarray of shape (k, d) containing the centroid means 4
            each cluster
    - clss: numpy.ndarray of shape (n,) containing the index of the cluster
            in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids
    C = initialize(X, k)
    if C is None:
        return None, None

    # Track the cluster assignments
    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):  # First (outside) loop
        # Make a copy to compare later
        old_C = C.copy()
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        new_clss = np.argmin(distances, axis=1)  # Shape (n,)

        clss = new_clss

        # Update centroids
        for j in range(k):  # Second (inside) loop
            cluster_points = X[clss == j]
            if len(cluster_points) == 0:
                # Reinitialize centroid if no points are assigned to the
                # cluster
                C[j] = np.random.uniform(low=X.min(axis=0),
                                         high=X.max(axis=0),
                                         size=(d,))
            else:
                C[j] = cluster_points.mean(axis=0)

        if np.allclose(old_C, C):
            clss = find_centroids(X, C)
            break

    return C, clss


def initialize(X, k):
    """
    Initialize K-means cluster centroids using uniform distribution.

    Args:
        X: numpy.ndarray of shape (n,d) containing dataset 4 clustering
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


def find_centroids(X, C):
    """find index of the closest centroid for each data point."""
    dist = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    return np.argmin(dist, axis=-1) # last
