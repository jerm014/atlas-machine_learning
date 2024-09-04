#!/usr/bin/env python3
""" documentation for the file goes here. """


def poly_derivative(poly):
    """
    documentatino for the function goes here.
    """
    if not poly or type(poly) is not list:
        return None

    res = []

    for o in range(1, len(poly)):
        res.append(o * poly[o])

    if len(res) == 0:
        res.append(0)

    return res
