#!/usr/bin/env python3
"""create_confusion_matrix function for project 2295 task 0"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix

    Args:
        labels (numpy.ndarray): correct
        logits (numpy.ndarray): predicted
    each arg is an ndarray of shape (m, classes) 
       where m is the number of examples
       and classes is the number of classes

    Returns:
        numpy.ndarray: A confusion matrix, that's the whole point.

    Credit:
        HT to nathan for reminding me that T transposes an ndarray.
    """
    return np.dot(labels.T, logits)
