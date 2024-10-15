#!/usr/bin/env python3
"""sensitivity function for project 2295 task 1"""

import numpy as np


def sensitivity(confusion):
    """
    Calculate the sensitivity of a confusion matrix

    Args:
        confusion (numpy.ndarray): confusion matrix (from task 0 for example!)

    Returns:
        float: sensitivity

    Notes from PLD:
        Sensitivity is the true positive rate.
        also called recall
        formula is tp / tp + fn

    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
