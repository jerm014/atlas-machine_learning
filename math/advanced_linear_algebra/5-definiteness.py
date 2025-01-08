#!/usr/bin/env python3
def determinant(matrix):
    # Check if input is a valid list of lists
    if not IsListOfLists(matrix):
        raise TypeError ("matrix must be a list of lists")

    if not IsSquareMatrix(matrix):
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1

    
    return

def IsListOfLists(param):
    if not isinstance(param, list):
        return False
    for item in param:
        if not isinstance(item, list):
            return False
    return True

def IsSquareMatrix(matrix):
   n = len(matrix)
   for row in matrix:
       if len(row) != n:
           return False
   return True