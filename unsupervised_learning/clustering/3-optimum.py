#!/usr/bin/env python3
"""Function to find optimum number of clusters by variance."""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Find optimum number of clusters by analyzing variance differences.

    Args:
        X: numpy.ndarray of shape (n, d) containing dataset
        kmin: minimum number of clusters to check (inclusive)
        kmax: maximum number of clusters to check (inclusive)
        iterations: maximum number of iterations for K-means

    Returns:
        results: list of K-means outputs for each cluster size
        d_vars: list of variance differences from smallest cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None:
        if not isinstance(kmax, int) or kmax <= 0:
            return None, None
        if kmin >= kmax:
            return None, None
    else:
        kmax = X.shape[0]
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None
        if k == kmin:
            first_var = var
        d_vars.append(first_var - var)

    return results, d_vars
