#!/usr/bin/env python3
"""Module containing normalize function"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Args:
    X (numpy.ndarray): Input matrix of shape (d, nx)
        d is the number of data points
        nx is the number of features
    m (numpy.ndarray): Mean of all features of X, shape (nx,)
    s (numpy.ndarray): Standard deviation of all features of X, shape (nx,)

    Returns:
    numpy.ndarray: The normalized X matrix
    """
    return (X - m) / s
