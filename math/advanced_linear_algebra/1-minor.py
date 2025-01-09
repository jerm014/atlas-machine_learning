#!/usr/bin/env python3
'''find the minor of a matrix'''

determinant = __import__('0-determinant').determinant
IsListOfLists = __import__('0-determinant').IsListOfLists
IsSquareMatrix = __import__('0-determinant').IsSquareMatrix

def minor(matrix):
    '''returns the minor of a matrix'''

    # Check if input is a valid list of lists
    if not IsListOfLists(matrix):
        raise TypeError('matrix must be a list of lists')

    if matrix == [[]]:
        return 1

    # Check if input is a square matrix
    if not IsSquareMatrix(matrix):
        raise ValueError('matrix must be a square matrix')



    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []

    

    return minor_matrix