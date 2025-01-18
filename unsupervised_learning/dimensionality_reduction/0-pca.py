#!/usr/bin/env python3
"""
Write a function def pca(X, var=0.95): that performs PCA on a dataset:

 - X is a numpy.ndarray of shape (n, d) where:
 - n is the number of data points
 - d is the number of dimensions in each point

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
    # Step 1: Do the SVD Decomposition. Ignore the first return value.                    
    _, s, Vh = np.linalg.svd(X, full_matrices=False)

    # Step 2: Calculate the Explained Variance Ratio
    explained_var = (s ** 2) / np.sum(s ** 2)

    # Step 3: Calculate Cumulative Sum of Variance Ratios
    cumulative_var = np.cumsum(explained_var)

    # Step 4: Calculate Number of Components (default var is 0.95)
    n_components = np.argmax(cumulative_var >= var) + 1

    # Step 5: Transpose and Truncate
    res = Vh.T[:, :n]

    return res