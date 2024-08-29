#!/usr/bin/env python3
"""
Write a function def np_matmul(mat1, mat2): that performs matrix
multiplication:

 * You can assume that mat1 and mat2 are numpy.ndarrays
 * You are not allowed to use any loops or conditional statements
 * You can assume that mat1 and mat2 are never empty
"""
import numpy as np


def np_matmul(mat1, mat2):
    return (mat1 @ mat2)
