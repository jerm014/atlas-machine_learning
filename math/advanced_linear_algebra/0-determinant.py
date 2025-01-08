#!/usr/bin/env python3
'''make determinant stuff'''


def determinant(matrix):
    '''calculate determinant of matrix'''

    # Check if input is a valid list of lists
    if not IsListOfLists(matrix):
        raise TypeError('matrix must be a list of lists')
        return

    if matrix == [[]]:
        return 1

    # Check if input is a square matrix
    if not IsSquareMatrix(matrix):
        raise ValueError('matrix must be a square matrix')
        return

    n = len(matrix)

    if n == 0:
        return 1

    if n == 1:
        return matrix[0][0]

    if n == 2:
        # det = (a * d) - (b * c)
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        minor = [[matrix[i][k] for k in range(n) if k != j]
                 for i in range(1, n)]
        # ((-1) ** j) makes the sign flip for every pass of j
        det += matrix[0][j] * ((-1) ** j) * determinant(minor)

    return det


def IsListOfLists(param):
    if not isinstance(param, list):
        return False
    for item in param:
        if not isinstance(item, list):
            return False
    return True


def IsSquareMatrix(matrix):
    n = len(matrix)
    if n == 0:
        return True
    for row in matrix:
        if len(row) != n:
            return False
    return True
