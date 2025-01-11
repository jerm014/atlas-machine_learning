#!/usr/bin/env python3
"""Project 2322 - Multivariate Probability - Task 1: Correlation"""
import numpy as np

ERR_TYPE = "C must be a numpy.ndarray"
ERR_SHAPE = "C must be a 2D square matrix"


def correlation(C):
    """
    Write a function def correlation(C): that calculates a correlation matrix:

    C is a numpy.ndarray of shape (d, d) containing a covariance matrix

    d is the number of dimensions

    If C is not a numpy.ndarray, raise a TypeError with the message C must be
    a numpy.ndarray

    If C does not have shape (d, d), raise a ValueError with the message C
    must be a 2D square matrix

    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """

    # Typechecking
    if not isinstance(C, np.ndarray):
        raise TypeError(ERR_TYPE)
    
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(ERR_SHAPE)

    std_devs = np.sqrt(np.diag(C))

    outer_std = np.outer(std_devs, std_devs)

    return C / outer_std
