#!/usr/bin/env python3
"""
Write a function def kmeans(X, k, iterations=1000): that performs K-means on a
dataset:

 - X is a numpy.ndarray of shape (n, d) containing the dataset
   - n is the number of data points
   - d is the number of dimensions for each data point
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

 - C is a numpy.ndarray of shape (k, d) containing the centroid means for each
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
    - C:    numpy.ndarray of shape (k, d) containing the centroid means for
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

    # Initialize centroids using a uniform distribution
    C = initialize(X, d)

    # Initialize previous centroids for comparison
    prev_C = np.zeros_like(C)

    for i in range(iterations):
        # Assign clusters: calculate the distance from each point to each
        # centroid
        distances = np.linalg.norm(
                                   X[:, np.newaxis, :] - C[np.newaxis, :, :],
                                   axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for j in range(k):
            if np.any(clss == j):
                C[j] = X[clss == j].mean(axis=0)
            else:
                # Reinitialize empty cluster centroid
                C[j] = initialize(X, d)

        # Check for convergence
        if np.allclose(C, prev_C):
            break

        prev_C = C.copy()

    return C, clss


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
