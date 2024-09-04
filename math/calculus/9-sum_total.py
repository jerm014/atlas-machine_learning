#!/usr/bin/env python3
""" place the documenation for the file here. """


def summation_i_squared(n):
    """
    place the documentation for the function here.
    """
    if type(n) is not int or n < 1:
        return None
    return (n*(n + 1) * (2*n + 1))//6
