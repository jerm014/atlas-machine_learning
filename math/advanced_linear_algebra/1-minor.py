#!/usr/bin/env python3
"""Project 2326 - Advanced Linear Algebra - Task 1: Minor"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Write a function def minor(matrix): that calculates the minor matrix of a
    matrix:

    matrix is a list of lists whose minor matrix should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists

    If matrix is not square or is empty, raise a ValueError with the
    message matrix must be a non-empty square matrix

    Returns: the minor matrix of matrix
    """

    if not isinstance(matrix, list) or not all(isinstance(row, list) for
                                               row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not all(len(row) == len(matrix) for row in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    new_matrix = [[0 for _ in range(len(matrix))]
                  for _ in range(len(matrix))]

    for r in range(len(matrix)):
        for c in range(len(matrix)):
            x = sub_matrix(matrix, r, c)
            if x == [[]]:
                return [[1]]
            new_matrix[r][c] = determinant(x)
    return new_matrix


def sub_matrix(matrix, r, c):
    """
    Take a matrix and remove a row and column.
    return the resulting matrix.
    """

    if len(matrix) == 1:
        return [[]]
    # Minor matrix excluding first row and current column
    return [[matrix[i][k] for k in range(len(matrix)) if k != c]
            for i in range(len(matrix)) if i != r]
