#!/usr/bin/env python3
"""Module for Normal distribution class."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution.

        Args:
            data (list, optional): List of data to estimate distribution.
            mean (float, optional): Mean of the distribution. Defaults to 0.
            stddev (float, optional): Standard deviation of the distribution.
                                      Defaults to 1.

        Raises:
            ValueError: If stddev <= 0 and data is None.
            TypeError: If data is not a list.
            ValueError: If data contains fewer than 2 values.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = (sum((x - self.mean) ** 2 for x in data) /
                           len(data)) ** 0.5
            self.stddev = float(self.stddev)

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value of z.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coefficient * e ** exponent

    def cdf(self, x):
        """
        Calculate the value of the CDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        erf = erf(self.z_score(x))
        if x >= 0 sign = 1 else sign = -1

        # Calculate the CDF
        return 0.5 * (1 + sign * erf)

def erf(x):
  """ approximation for erf """
    # Constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    if x >= 0 sign = 1 else sign = -1
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
               exp(-x * x)

    return sign * y

def exp(x):
    """
    Taylor series approximation for e^x
    """
    n = 20  # Number of terms in the series
    result = 1.0
    term = 1.0
    for i in range(1, n):
        term *= x / i
        result += term
    return result
