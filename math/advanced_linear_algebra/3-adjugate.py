#!/usr/bin/env python3
"""Project 2326 - Advanced Linear Algebra - Task 3: Adjugate"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Write a function def adjugate(matrix): that calculates the adjugate matrix
    of a matrix:

    matrix is a list of lists whose adjugate matrix should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists

    If matrix is not square or is empty, raise a ValueError with the message
    matrix must be a non-empty square matrix

    Returns: the adjugate matrix of matrix
    """

    cofactor_matrix = cofactor(matrix)

    result = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for r in range(len(matrix)):
        for c in range(len(matrix)):
            result[c][r] = cofactor_matrix[r][c]
    return result
