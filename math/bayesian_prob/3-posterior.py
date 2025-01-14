#!/usr/bin/env python3
"""
Based on 2-marginal.py, write a function def posterior(x, n, P, Pr): that
calculates the posterior probability for the various hypothetical
probabilities of developing severe side effects given the data:

 - x  is the number of patients that develop severe side effects
 - n  is the total number of patients observed
 - P  is a 1D numpy.ndarray containing the various hypothetical
      probabilities of developing severe side effects
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

All exceptions should be raised in the above order

Returns: the posterior probability of each probability in P given x and n,
         respectively

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


class Bayes:
    """Class for calculating Bayesian probabilities"""

    def __init__(self, x, n, P, Pr=None):
        """Initialize Bayes calculator with input parameters

        Args:
            x: Number of successes
            n: Number of trials
            P: Array of probabilities
            Pr: Array of prior probabilities (optional)
        """
        self._validate_inputs(x, n, P, Pr)
        self.x = x
        self.n = n
        self.P = P
        self.Pr = Pr

    def _validate_inputs(self, x, n, P, Pr=None):
        """Validate all input parameters"""
        if not isinstance(n, int) or n <= 0:
            raise ValueError(E1)
        if not isinstance(x, int) or x < 0:
            raise ValueError(E2)
        if x > n:
            raise ValueError(E3)
        if not isinstance(P, np.ndarray) or P.ndim != 1:
            raise TypeError(E4)
        if not np.all((P >= 0) & (P <= 1)):
            raise ValueError(E6)

        if Pr is not None:
            if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
                raise TypeError(E5)
            if not np.all((Pr >= 0) & (Pr <= 1)):
                raise ValueError(E7)
            if not np.isclose(np.sum(Pr), 1):
                raise ValueError(E8)

    def likelihood(self):
        """Calculate the likelihood of obtaining the data"""
        return (_fact(self.n) / (_fact(self.x) * _fact(self.n - self.x)) *
                self.P ** self.x * (1 - self.P) ** (self.n - self.x))

    def intersection(self):
        """Calculate the intersection of obtaining the data"""
        return self.Pr * self.likelihood()

    def marginal(self):
        """Calculate the marginal probability of obtaining the data"""
        return np.sum(self.intersection())

    def posterior(self):
        """Calculate the posterior probability"""
        return self.intersection() / self.marginal()


def likelihood(x, n, P):
    """External function to calculate likelihood"""
    bayes = Bayes(x, n, P)
    return bayes.likelihood()


def intersection(x, n, P, Pr):
    """External function to calculate intersection"""
    bayes = Bayes(x, n, P, Pr)
    return bayes.intersection()


def marginal(x, n, P, Pr):
    """External function to calculate marginal probability"""
    bayes = Bayes(x, n, P, Pr)
    return bayes.marginal()


def posterior(x, n, P, Pr):
    """External function to calculate posterior probability"""
    bayes = Bayes(x, n, P, Pr)
    return bayes.posterior()


def _fact(n):
    """Calculate the factorial of a number"""
    return np.math.factorial(n)
