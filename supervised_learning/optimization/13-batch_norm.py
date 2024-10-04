#!/usr/bin/env python3
"""Module containing batch_norm function"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Args:
    Z (numpy.ndarray): Array of shape (m, n) to be normalized.
        m is the number of data points
        n is the number of features in Z
    gamma (numpy.ndarray): Array of shape (1, n) containing the scales for
        batch normalization.
    beta (numpy.ndarray): Array of shape (1, n) containing the offsets for
        batch normalization.
    epsilon (float): Small number used to avoid division by zero.

    Returns:
    numpy.ndarray: The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_scaled = gamma * Z_norm + beta
    return Z_scaled
