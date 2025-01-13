#!/usr/bin/env python3
"""
Based on 0-likelihood.py, write a function def intersection(x, n, P, Pr): that
calculates the intersection of obtaining this data with the various
hypothetical probabilities:

 - x  is the number of patients that develop severe side effects
 - n  is the total number of patients observed
 - P  is a 1D numpy.ndarray containing the various hypothetical probabilities
      of developing severe side effects
 - Pr is a 1D numpy.ndarray containing the prior beliefs of P

If n is not a positive integer, raise a ValueError with the message n must be
a positive integer

If x is not an integer that is greater than or equal to 0, raise a ValueError
with the message x must be an integer that is greater than or equal to 0

If x is greater than n, raise a ValueError with the message x cannot be
greater than n

If P is not a 1D numpy.ndarray, raise a TypeError with the message P must be a
1D numpy.ndarray

If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError with
the message Pr must be a numpy.ndarray with the same shape as P

If any value in P or Pr is not in the range [0, 1], raise a ValueError with
the message All values in {P} must be in the range [0, 1] where {P} is the
incorrect variable

If Pr does not sum to 1, raise a ValueError with the message Pr must sum to 1
Hint: use numpy.isclose

All exceptions should be raised in the above order

Returns: a 1D numpy.ndarray containing the intersection of obtaining x and n
         with each probability in P, respectively

"""
import numpy as np
E1 = "n must be a positive integer"
E2 = "x must be an integer that is greater than or equal to 0"
E3 = "x cannot be greater than n"
E4 = "P must be a 1D numpy.ndarray"
E5 = "Pr must be a numpy.ndarray with the same shape as P"
E6 = "All values in {} must be in the range [0, 1]"
E7 = "Pr must sum to 1"


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this data with the various
    hypothetical probabilities"""

    # If n is not a positive integer, raise a ValueError
    if not isinstance(n, int) or n <= 0:
        raise ValueError(E1)

    # If x is not an integer that is greater than or equal to 0, raise a
    # ValueError
    if not isinstance(x, int) or x < 0:
        raise ValueError(E2)

    # If x is greater than n, raise a ValueError
    if x > n:
        raise ValueError(E3)

    # If P is not a 1D numpy.ndarray, raise a TypeError
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError(E4)

    # If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(E5)

    # If any value in P or Pr is not in the range [0, 1], raise a ValueError
    if not np.all((P >= 0) & (P <= 1)) or not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError(E6.format(P if np.all((P >= 0) & (P <= 1)) else Pr))

    # If Pr does not sum to 1, raise a ValueError -[Hint: use numpy.isclose]-
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError(E7)

    a = Pr * factorial(n)
    b = factorial(x) * factorial(n - x)
    c = P ** x * (1 - P) ** (n - x)

    return a / b * c


def factorial(n):
    """
    Calculate the factorial of a non-negative integer using numpy.

    Args:    n: A non-negative integer to calculate factorial for

    Returns: The factorial of n
    """
    return np.math.factorial(n)
