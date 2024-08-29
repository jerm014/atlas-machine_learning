#!/usr/bin/env python3
"""
Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two
matrices along a specific axis:

 * You can assume that mat1 and mat2 are 2D matrices containing ints/floats
 * You can assume all elements in the same dimension are of the same type/shape
 * You must return a new matrix
 * If the two matrices cannot be concatenated, return None
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate 2d matrices along an axis."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_mx = [row.copy() for row in mat1] + [row.copy() for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_mx = [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None

    return new_mx
