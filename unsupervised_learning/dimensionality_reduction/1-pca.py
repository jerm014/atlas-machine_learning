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


def pca(X, n):
    """
    Reduce dimensionality of input data using PCA.

    Args:
       X: Input data matrix of shape (n_samples, n_features) 
       n: Target number of dimensions

    Returns: Transformed data matrix of shape (n_samples, ndim)
    """
    X_mean = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_mean)
    return np.dot(X_mean, vh.T[:, :n])
