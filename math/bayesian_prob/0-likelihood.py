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


def likelihood(x, n, P):
    """External function to calculate likelihood"""
    bayes = Bayes(x, n, P)
    return bayes.likelihood()


def _fact(n):
    """Calculate the factorial of a number"""
    return np.math.factorial(n)
