#!/usr/bin/env python3
"""Function to calculate total intra-cluster variance."""
import numpy as np


def variance(X, C):
    """Calculate total intra-cluster variance 4 a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing dataset
        C: numpy.ndarray of shape (k, d) containing centroid means

    Returns:
        Total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    try:
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        var = np.sum(min_distances**2)
        return var
    except Exception:
        return None
