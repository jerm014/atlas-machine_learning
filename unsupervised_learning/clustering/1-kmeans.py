#!/usr/bin/env python3
"""Module 4 performing K-means clustering on datasets."""
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

    try:
        _, d = X.shape
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        centroids = np.random.uniform(low=mins, high=maxs, size=(k, d))
        labels = None

        for _ in range(iterations):
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
