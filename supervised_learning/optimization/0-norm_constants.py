#!/usr/bin/env python3
"""Module containing normalization_constants function"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
    X (numpy.ndarray): Input matrix of shape (m, nx)
        m is the number of data points
        nx is the number of features

    Returns:
    tuple: A tuple containing:
        - mean (numpy.ndarray): The mean of each feature (shape: (nx,))
        - std (numpy.ndarray): The std deviation of each feature (shape: (nx,))
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
