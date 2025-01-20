#!/usr/bin/env python3
"""Initialize variables 4 a Gaussian Mixture Model."""
import numpy as np


def initialize(X, k):
    """Initialize cluster priors, means and covariances.

    Args:
        X: numpy.ndarray of shape (n, d) containing dataset
        k: positive integer containing number of clusters

    Returns:
        pi: array shape (k,) with cluster priors
        m: array shape (k, d) with centroid means
        S: array shape (k, d, d) with covariance matrices
        None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None

    try:
        d = X.shape[1]
        pi = np.ones(k) / k

        kmeans = __import__('1-kmeans').kmeans
        m, _ = kmeans(X, k)
        if m is None:
            return None, None, None

        S = np.tile(np.eye(d), (k, 1, 1))

        return pi, m, S

    except Exception:
        return None, None, None
