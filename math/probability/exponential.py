#!/usr/bin/env python3
"""Module for Exponential distribution class."""


class Exponential:
    """Represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution.

        Args:
            data (list, optional): List of data to estimate distribution.
            lambtha (float, optional): Expected number of occurrences in a
                given time frame. Defaults to 1.

        Raises:
            ValueError: If lambtha <= 0 and data is None.
            TypeError: If data is not a list.
            ValueError: If data contains fewer than 2 values.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.
        """
        if x < 0:
            return 0

        # do not i m p o r t math
        e = 2.7182818285

        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculate the value of the CDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The CDF value for x.
        """
        if x < 0:
            return 0

        # do not i m p o r t math
        e = 2.7182818285

        return 1 - e ** (-self.lambtha * x)
