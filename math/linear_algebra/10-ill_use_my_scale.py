#!/usr/bin/env python3
"""
Write a function def np_shape(matrix): that calculates the shape of a
numpy.ndarray:

 * You are not allowed to use any loops or conditional statements
 * You are not allowed to use try/except statements
 * The shape should be returned as a tuple of integers
"""
import numpy as np


def np_shape(matrix):
    """
    Calculate the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The input numpy array.

    Returns:
        tuple: A tuple of integers representing the shape of the array.

    Note:
        This function does not use loops, conditional statements, or
        try/except.
    """
    return tuple(map(len, (matrix,) + matrix))
