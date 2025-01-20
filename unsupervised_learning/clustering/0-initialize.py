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
