#!/usr/bin/env python3
"""Module for initializing K-means clustering centroids."""
import numpy as np


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
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        return np.random.uniform(low=mins, high=maxs, size=(k, X.shape[1]))
    except Exception:
        return None
