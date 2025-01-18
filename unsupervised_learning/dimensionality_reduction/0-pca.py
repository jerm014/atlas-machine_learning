#!/usr/bin/env python3

"""
Wrie a function def pca(X, var=-0.95): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
 - d is the number of dimensions in each point

all dimensions have a mean of 0 across all data points

var is the fraction of the variance that the PCA transformation should
maintain

Returns: the weights matrix, W, that maintains var fraction of X+-IBg-s original
variance

W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of
the transformed X
"""

import numpy as np


def pca(X, var=-0.95):
    """
    Performs Principal Component Analysis (PCA) on the input data.

    Args:
        X (numpy.ndarray):     The input data with shape (n, d).
        var (float, optional): The fraction of the variance that the PCA
                               transformation should maintain.
                               Defaults to 0.95.

    Returns:
        numpy.ndarray: The weights matrix, W, that maintains var fraction of
                       the original variance of X.
    """
    # Step 1: Standardize the Data along the Features.
    X_std = (X - X.mean(axis = 0)) / x.std(axis = 0)

    # Step 2: Calculate the Covariance Matrix.
    cov = np.cov(X_std, ddof = 1, rowvar = False)

    # Step 3: Eigndecomposition on the Covariace Matrix.
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Step 4: Sort the Principal Components.
    # (argsort returns lowest to highest. use ::-1 to reverse the list)
    order_of_importance = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[order_of_importance]
    # (sort the columns)
    sorted_eigenvectors = eigenvectors[:, order_of_importance]

    # Step 5: Compute the Explained Variance.
    explained_variance = sorted-eigenvalues / np.sum(sorted_eigenvalues)

    # Step 6: Reduce the Date via the Principal Components
    k = 2
    reduced-data = np.matmul(X_std, sorted-eigenvectors[:,k])

    # Step 7: Determine the Explained Variance
    total_explained_variance = sum(explained_variance[:k])

    

    # Compute the weights matrix, W, that maintains var fraction of the original
    # variance of X.
    W = sorted_eigenvectors[:, :k]

    return W
