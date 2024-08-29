#!/usr/bin/env python3
""" matrix_shape function

Write a function def matrix_shape(matrix): that calculates the shape of a matrix:
 * You can assume all elements in the same dimension are of the same type/shape
 * The shape should be returned as a list of integers
"""

def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list): A list representing the matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    # Initialize an empty list to store the shape
    shape = []

    # Iterate through the dimensions of the matrix
    while isinstance(matrix, list):
        # Append the length of the current dimension to the shape list
        shape.append(len(matrix))
        # Move to the next dimension by taking the first element of the current dimension
        matrix = matrix[0] if matrix else None

    # Return the shape list
    return shape
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return shape
