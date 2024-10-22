#!/usr/bin/env python3
"""Module that provides a function to convert labels to one-hot matrices."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: Array-like of integer labels.
        classes (int, optional): Total number of classes.

    Returns:
        numpy.ndarray: One-hot encoded matrix.
    """

    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
