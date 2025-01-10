#!/usr/bin/env python3
'''Project 2326 - Advanced Linear Algebra - Task 5: Definiteness'''
import numpy as np


def definiteness(matrix):
    """
    Write a function def definiteness(matrix): that calculates the
    definiteness of a matrix:

    matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
    calculated

    If matrix is not a numpy.ndarray, raise a TypeError with the message
    matrix must be a numpy.ndarray

    If matrix is not a valid matrix, return None

    Return: the string Positive definite, Positive semi-definite, Negative
            semi-definite, Negative definite, or Indefinite if the matrix is
            positive definite, positive semi-definite, negative semi-definite,
            negative definite OR indefinite, respectively

    If matrix does not fit any of the above categories, return None

    You may import numpy as np
    """

    # First check if matrix is symmetric
    # or don't?

    # Get eigenvalues
    eigenvals = safe_eigvals(matrix)

    try:
        checks = np.array([
            np.all(eigenvals > 0),     # Positive definite
            np.all(eigenvals >= 0),    # Positive semidefinite
            np.all(eigenvals < 0),     # Negative definite
            np.all(eigenvals <= 0),    # Negative semidefinite
            1                          # Indefinite (default)
        ])
    except e as Exception:
        return None

    # Create array of possible results
    results = np.array([
        'Positive definite',
        'Positive semi-definite',
        'Negative definite',
        'Negative semi-definite',
        'Indefinite'
    ])

    # Return first matching result
    return results[checks.argmax()]


def safe_eigvals(matrix):
    try:
        return np.linalg.eigvals(matrix)
    except e as Exception:
        return None
