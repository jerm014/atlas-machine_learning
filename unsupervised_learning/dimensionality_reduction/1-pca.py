#!/usr/bin/env python3
"""
Write a function def pca(X, ndim): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
 - d is the number of dimensions in each point

ndim is the new dimensionality of the transformed X

Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
version of X
"""

import numpy as np


def pca(X, ndim):
    """
    Do the thing.
    """
    X_mean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_mean)
    W = vh.T[:, :ndim]
    return np.dot(X_mean, W)
