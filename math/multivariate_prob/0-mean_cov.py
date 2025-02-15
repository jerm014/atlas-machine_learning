#!/usr/bin/env python3
"""Project 2322 - Multivariate Probability - Task 0: Mean and Covariance"""
import numpy as np


def mean_cov(X):
    """
    Write a function def mean_cov(X): that calculates the mean and covariance
    of a data set:

      *  X is a numpy.ndarray of shape (n, d) containing the data set:
        -  n is the number of data points
        -  d is the number of dimensions in each data point

      *  If X is not a 2D numpy.ndarray, raise a TypeError with the message X
         must be a 2D numpy.ndarray

      *  If n is less than 2, raise a ValueError with the message X must
         contain multiple data points

    Returns: mean, cov:

      *  mean is a numpy.ndarray of shape (1, d) containing the mean of the
         data set
      *  cov is a numpy.ndarray of shape (d, d) containing the covariance
         matrix of the data set

    You are not allowed to use the function numpy.cov
    """

    # Check for valid X
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n, d = X.shape

    if n < 2:
        raise ValueError('X must contain multiple data points')

    # Calculate the mean of the dataset
    Xmean = np.mean(X, axis=0).reshape(1, -1)

    # Calculate covariance matrix
    # Formula: cov = (X - mean)^T @ (X - mean) / (n-1)
    Xcovariance = np.matmul((X - Xmean).T, X - Xmean) / (n - 1)

    return Xmean, Xcovariance
