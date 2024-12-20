#!/usr/bin/env python3
"""
Binomial Distribution Module

This module provides a Binomial class for representing and working with
binomial distributions. It allows for the creation of binomial distributions
either by specifying parameters directly or by estimating them from data.

Classes:
    Binomial: Represents a binomial distribution.

Usage:
    from binomial i*mport Binomial

    # Create a binomial distribution with specified parameters
    b1 = Binomial(n=10, p=0.3)

    # Create a binomial distribution from data
    data = [2, 3, 1, 4, 2, 3, 2, 1, 3, 2]
    b2 = Binomial(data=data)

Note:
    This module does not depend on any external libraries and uses only
    built-in Python functions.
"""


class Binomial:
    """
    Represents a binomial distribution.

    Attributes:
        n (int): Number of Bernoulli trials.
        p (float): Probability of success for each trial.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        Args:
            data (list, optional): Data to estimate the distribution.
            n (int, optional): Number of Bernoulli trials. Defaults to 1.
            p (float, optional): Probability of success. Defaults to 0.5.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If data has less than two values, n is not positive,
                        or p is not a valid probability.
        """
        if data is None:
            self._validate_n_p(n, p)
            self.n = int(n)
            self.p = float(p)
        else:
            self._estimate_from_data(data)

    def _validate_n_p(self, n, p):
        """Validate n and p values."""
        if n <= 0:
            raise ValueError("n must be a positive value")
        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")

    def _estimate_from_data(self, data):
        """Estimate n and p from given data."""
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")

        self.p = sum(data) / len(data)
        self.n = round(max(data) / self.p)
        self.p = sum(data) / (self.n * len(data))

    def _factorial(self, x):
        """Calculate the factorial of x."""
        if x == 0 or x == 1:
            return 1
        return (x * self._factorial(x - 1))

    def _combination(self, n, k):
        """Calculate the number of ways to choose k items from n items."""
        return (self._factorial(n) //
                (self._factorial(k) * self._factorial(n - k)))

    def pmf(self, k):
        """
        Calculate the Probability Mass Function (PMF) for k successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: The PMF value for k successes.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return (self._combination(self.n, k) * (self.p ** k) *
                ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """
        Calculate the Cumulative Distribution Function (CDF) for k successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: The CDF value for k successes.
        """
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1
        return sum(self.pmf(i) for i in range(k + 1))
