#!/usr/bin/env python3
""" Module containing the one_hot_encode function """

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): Array with shape (m,) containing numeric class
                           labels
        classes (int):     Maximum number of classes found in Y

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m), or None
        on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except IndexError:
        return None
