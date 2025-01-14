#!/usr/bin/env python3
"""
Based on 1-intersection.py, write a function def marginal(x, n, P, Pr): that
calculates the marginal probability of obtaining the data:

 - x  is the number of patients that develop severe side effects
 - n  is the total number of patients observed
 - P  is a 1D numpy.ndarray containing the various hypothetical probabilities
      of patients developing severe side effects
 - Pr is a 1D numpy.ndarray containing the prior beliefs about P
If n is not a positive integer, raise a ValueError with the message n must be
a positive integer

If x is not an integer that is greater than or equal to 0, raise a ValueError
with the message x must be an integer that is greater than or equal to 0

If x is greater than n, raise a ValueError with the message x cannot be
greater than n

If P is not a 1D numpy.ndarray, raise a TypeError with the message P must be
a 1D numpy.ndarray

If Pr is not a numpy.ndarray with the same shape as P, raise a TypeError with
the message Pr must be a numpy.ndarray with the same shape as P

If any value in P or Pr is not in the range [0, 1], raise a ValueError with
the message All values in {P} must be in the range [0, 1] where {P} is the
incorrect variable

If Pr does not sum to 1, raise a ValueError with the message Pr must sum to 1

All exceptions should be raised in the above order

Returns: the marginal probability of obtaining x and n

"""
import numpy as np

E1 = "n must be a positive integer"
E2 = "x must be an integer that is greater than or equal to 0"
E3 = "x cannot be greater than n"
E4 = "P must be a 1D numpy.ndarray"
E5 = "Pr must be a numpy.ndarray with the same shape as P"
E6 = "All values in P must be in the range [0, 1]"
E7 = "All values in Pr must be in the range [0, 1]"
E8 = "Pr must sum to 1"


def marginal(x, n, P, Pr):
    """Calculate the marginal probability of obtaining the data"""
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
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError(E6)

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError(E7)

    # If Pr does not sum to 1, raise a ValueError -[Hint: use numpy.isclose]-
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError(E8)

    return np.sum(intersection(x, n, P, Pr))


def intersection(x, n, P, Pr):
    """Calculate the intersection of obtaining the data"""

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
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError(E6)

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError(E7)

    # If Pr does not sum to 1, raise a ValueError -[Hint: use numpy.isclose]-
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError(E8)

    return Pr * likelihood(x, n, P)


def likelihood(x, n, P):
    """Calculate the likelihood of obtaining the data"""
    if type(n) is not int or n <= 0:
        raise ValueError(E1)
    if type(x) is not int or x < 0:
        raise ValueError(E2)
    if x > n:
        raise ValueError(E3)
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError(E4)
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError(E6)

    return fact(n) / (fact(x) * fact(n - x)) * P ** x * (1 - P) ** (n - x)


def fact(n):
    """Calculate the factorial of a non-negative integer using numpy"""
    return np.math.factorial(n)
