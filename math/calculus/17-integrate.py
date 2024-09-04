#!/usr/bin/env python3
""" this is where file documenation goes. """


def poly_integral(poly, C=0):
    """
    write better documentation.
    """
    if not poly or type(poly) is not list:
        return None

    if type(C) is not int:
        return None

    if sum(poly):
        out = [fint(c/(i + 1)) for i, c in enumerate(poly)]
    else:
        out = []

    out.insert(0, C)

    return out


def fint(num):
    """ function documentation here """
    return int(num) if int(num) == num else num
