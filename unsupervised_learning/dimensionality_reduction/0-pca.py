#!/usr/bin/env python3

"""
Wrie a function def pca(X, var=-0.95): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
 - d is the number of dimensions in each point

all dimensions have a mean of 0 across all data points

var is the fraction of the variance that the PCA transformation should
maintain

Returns: the weights matrix, W, that maintains var fraction of X+IBg-s original
variance

W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of
the transformed X
"""

import numpy as np


def pca(X, var=-0.95):
    """
    Performs Principal Component Analysis (PCA) on the input data.

    Args:
        X (numpy.ndarray): The input data with shape (n, d).
        var (float, optional): The fraction of the variance that the PCA
            transformation should maintain. Defaults to 0.95.

    Returns:
        numpy.ndarray: The weights matrix, W, that maintains var fraction of
            the original variance of X.
    """
    # Center the data by subtracting the mean
    X_mean = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov = np.cov(X_mean.T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvalues and eigenvectors in descending order of eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the cumulative explained variance ratio
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Find the number of dimensions that maintain the desired variance
    nd = np.argmax(cum_var >= var) + 1

    # Compute the weights matrix, W, that maintains var fraction of the original variance
    W = eigenvectors[:, :nd]

    return W
