#!/usr/bin/env python3
"""
Write a function def pca(X, var=0.95): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
s - d is the number of dimensions in each point

all dimensions have a mean of 0 across all data points

var is the fraction of the variance that the PCA transformation should
maintain

Returns: the weights matrix, W, that maintains var fraction of X‘s original
variance

W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of
the transformed X
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset while maintaining a specified fraction of
    variance.

    Parameters:
    X: numpy.ndarray of shape (n, d) where:
       - n is the number of data points
       - d is the number of dimensions in each point
    var: float, fraction of variance to maintain (default: 0.95)

    Returns:
    W: numpy.ndarray of shape (d, nd) where nd is the new dimensionality
       This is the transformation matrix that maintains var fraction of X's
       variance
    """
    # Calculate covariance matrix
    # Since X is already centered (mean=0), we can directly compute covariance
    covariance_matrix = np.cov(X.T)

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate explained variance ratios
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    # Find number of components needed to maintain desired variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= var) + 1

    # Select the first n_components eigenvectors
    W = eigenvectors[:, :n_components]

    return W
