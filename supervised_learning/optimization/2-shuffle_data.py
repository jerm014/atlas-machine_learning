#!/usr/bin/env python3
"""Module containing shuffle_data function"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
    X (numpy.ndarray): First input matrix of shape (m, nx)
        m is the number of data points
        nx is the number of features in X
    Y (numpy.ndarray): Second input matrix of shape (m, ny)
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns:
    tuple: A tuple containing:
        - X_shuffled (numpy.ndarray): The shuffled X matrix
        - Y_shuffled (numpy.ndarray): The shuffled Y matrix
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
