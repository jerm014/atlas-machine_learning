#!/usr/bin/env python3
"""
Write a function def matrix_transpose(matrix): that returns the transpose of a
2D matrix, matrix:

* You must return a new matrix
* You can assume that matrix is never empty
* You can assume all elements in the same dimension are of the same type/shape
"""

def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The input 2D matrix to be transposed.

    Returns:
        list of lists: A new matrix that is the transpose of the input matrix.

    Raises:
        IndexError: If the input matrix is empty.

    Note:
        - This function assumes that the input matrix is not empty.
        - It also assumes all elements in the same dimension are of the
          same type/shape.
    """
    # Get the number of rows and columns in the original matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matrix with swapped dimensions
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]

    # Fill the new matrix with transposed elements
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed

