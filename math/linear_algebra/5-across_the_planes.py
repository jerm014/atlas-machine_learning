#!/usr/bin/env python3
"""
Write a function def add_matrices2D(mat1, mat2): that adds two matrices
element-wise:

 * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
 * You can assume all elements in the same dimension are of the same type/shape
 * You must return a new matrix
 * If mat1 and mat2 are not the same shape, return None
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list): The first 2D matrix.
        mat2 (list): The second 2D matrix.

    Returns:
        list: A new 2D matrix containing the element-wise sum of mat1 and mat2.
              If mat1 and mat2 are not the same shape, returns None.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
