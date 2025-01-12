#!/usr/bin/env python3
"""
You are conducting a study on a revolutionary cancer drug and are looking to
find the probability that a patient who takes this drug will develop severe
side effects. During your trials, n patients take the drug and x patients
develop severe side effects. You can assume that x follows a binomial
distribution.

Write a function def likelihood(x, n, P): that calculates the likelihood of
obtaining this data given various hypothetical probabilities of developing
severe side effects:

 - x is the number of patients that develop severe side effects
 - n is the total number of patients observed
 - P is a 1D numpy.ndarray containing the various hypothetical probabilities
     of developing severe side effects

If n is not a positive integer, raise a ValueError with the message n must be
a positive integer

If x is not an integer that is greater than or equal to 0, raise a ValueError
with the message x must be an integer that is greater than or equal to 0

If x is greater than n, raise a ValueError with the message x cannot be
greater than n

If P is not a 1D numpy.ndarray, raise a TypeError with the message P must be a
1D numpy.ndarray

If any value in P is not in the range [0, 1], raise a ValueError with the
message All values in P must be in the range [0, 1]

Returns: a 1D numpy.ndarray containing the likelihood of obtaining the data,
x and n, for each probability in P, respectively
"""
import numpy as np

E1 = "n must be a positive integer"
E2 = "x must be an integer that is greater than or equal to 0"
E3 = "x cannot be greater than n"
E4 = "P must be a 1D numpy.ndarray"
E5 = "All values in P must be in the range [0, 1]"


def likelihood(x, n, P):
    """calculate the likelihood of obtaining this data?"""
    if type(n) is not int or n <= 0:
        raise ValueError(E1)
    if type(x) is not int or x < 0:
        raise ValueError(E2)
    if x > n:
        raise ValueError(E3)
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError(E4)
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError(E5)

    ret = factorial(n) / (factorial(x) * factorial(n - x))
    return ret * P ** x * (1 - P) ** (n - x)


def factorial(n):
    """
    Calculate the factorial of a non-negative integer using numpy.

    Args:    n: A non-negative integer to calculate factorial for

    Returns: The factorial of n
    """
    return np.math.factorial(n)
