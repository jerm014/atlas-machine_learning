#!/usr/bin/env python3
"""Module containing the one_hot_decode function"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): One-hot encoded array with shape (classes, m)

    Returns:
        numpy.ndarray: Array with shape (m,) containing the numeric labels for
        each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
