#!/usr/bin/env python3
"""precision function for project 2295 task 2"""

import numpy as np


def precision(confusion):
    """
    Calculates precision for each class in a confusion matrix
    
    Args:
      confusion is a confusion numpy.ndarray of shape (classes, classes)
        row indices represent the correct labels
        column indices represent the predicted labels
        classes is the number of classes

    Returns:
      numpy.ndarray of shape (classes,) containing precision of each class

    Notes from PLD:
      Precision measures the proportion of positive identifications that were
      actually correct. In other words, out of all the instances that our model
      labeled as positive, how many were *actually* positive?
    """
    return np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1e-7)
