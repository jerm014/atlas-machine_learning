#!/usr/bin/env python3
"""Project 2326 - Advanced Linear Algebra - Task 0: Determinant"""


def determinant(matrix):
    """
    Write a function def determinant(matrix): that calculates the determinant
    of a matrix:

    matrix is a list of lists whose determinant should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists

    If matrix is not square, raise a ValueError with the message matrix must
    be a square matrix

    The list [[]] represents a 0x0 matrix

    Returns: the determinant of matrix
    """

    # Is matrix a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for
                                               row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    # 2x2 matrix - this is the work!
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Laplace expansion along first row - recusion, baby
    det = 0
    for j in range(len(matrix)):
        # Minor matrix excluding row 0 and current column
        minor = [[matrix[i][k] for k in range(len(matrix)) if k != j]
                 for i in range(1, len(matrix))]
        # Adds positive or negative to determinant by raising -1 to j
        det += matrix[0][j] * ((-1) ** j) * determinant(minor)

    return det
