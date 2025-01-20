#!/usr/bin/env python3
"""Module for performing K-means clustering."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering on a dataset.

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

    try:
        n, d = X.shape
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        C = np.random.uniform(low=mins, high=maxs, size=(k, d))

        for i in range(iterations):
            # Calculate distances between points and centroids
            distances = np.sqrt(((X[:, np.newaxis] - C) ** 2).sum(axis=2))

            # Assign points to nearest centroid
            clss = np.argmin(distances, axis=1)

            # Store old centroids to check for convergence
            old_centroids = C.copy()

            # Update centroids
            for j in range(k):
                points = X[clss == j]
                if len(points) == 0:
                    C[j] = np.random.uniform(low=mins, high=maxs, size=d)
                else:
                    C[j] = points.mean(axis=0)

            # Check for convergence
            if np.all(old_centroids == C):
                break

        return C, clss

    except Exception:
        return None, None
