#!/usr/bin/env python3
"""Project 2326 - Advanced Linear Algebra - Task 4: Inverse"""

determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Write a function def inverse(matrix): that calculates the inverse of a
    matrix:

    matrix is a list of lists whose inverse should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists

    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the inverse of matrix, or None if matrix is singular
    """

    d = determinant(matrix)
    if d == 0:
        return None

    adjugate_matrix = adjugate(matrix)

    result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for r in range(len(matrix)):
        for c in range(len(matrix)):
            result[r][c] = adjugate_matrix[r][c] / d
    return result
