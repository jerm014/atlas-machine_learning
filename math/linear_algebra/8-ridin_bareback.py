#!/usr/bin/env python3
"""
Write a function def mat_mul(mat1, mat2): that performs matrix multiplication:

 * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
 * You can assume all elements in the same dimension are of the same type/shape
 * You must return a new matrix
 * If the two matrices cannot be multiplied, return None
"""


def mat_mul(mat1, mat2):
    """
    Perform matrix multiplication of two 2D matrices.

    Args:
        mat1 (list of lists): The first input matrix.
        mat2 (list of lists): The second input matrix.

    Returns:
        list of lists or None: A new matrix that is the product of mat1 and mat2,
                               or None if the matrices cannot be multiplied.

    Note:
        - Assumes mat1 and mat2 are 2D matrices containing ints/floats.
        - Assumes all elements in the same dimension are of the same type/shape.
    """
    # Check if matrices can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Get dimensions
    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    # Initialize result matrix
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Perform matrix multiplication
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
