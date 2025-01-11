#!/usr/bin/env python3
import numpy as np

def mean_cov(X):
    """
    Calculate the mean and covariance of a dataset.

    Args:
        X (numpy.ndarray): The input dataset.

    Returns:
        tuple: A tuple containing the mean and covariance of the dataset.
    """

    # Check for valid X
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    if  len(X.shape) < 2:
        raise ValueError('X must contain multiple data points')

    # Calculate the mean of the dataset
    Xmean = np.mean(X, axis=0).reshape(1, -1)

    n, _ = X.shape

    # Calculate covariance matrix
    # Formula: cov = (X - mean)^T @ (X - mean) / (n-1)
    Xcovariance = np.matmul(X - Xmean.T, X - Xmean) / (n - 1)

    return Xmean, Xcovariance