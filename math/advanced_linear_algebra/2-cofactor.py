#!/usr/bin/env python3
"""Project 2326 - Advanced Linear Algebra - Task 2: Cofactor"""
determinant = __import__('0-determinant').determinant
sub_matrix = __import__('1-minor').sub_matrix


def cofactor(matrix):
    """
    Write a function def cofactor(matrix): that calculates the cofactor matrix
    of a matrix:

    matrix is a list of lists whose cofactor matrix should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists

    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the cofactor matrix of matrix
    """

    result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]

    for r in range(len(matrix)):
        for c in range(len(matrix)):
            x = sub_matrix(matrix, r, c)
            if x == [[]]:
                return [[1]]
            result[r][c] = determinant(x) * ((-1) ** (r + c))
    return result
