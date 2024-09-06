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
        Calculate the value of the CDF for a given x-value, using the Hart
        approximation.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        z = self.z_score(x)
        
        # Constants for approximation
        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        pi = 3.1415926536
        e = 2.7182818285
        
        t = 1 / (1 + b0 * abs(z))
        
        # Approximation formula
        cdf = 1 - (1 / ((2 * pi) ** 0.5)) * \
              e ** (-0.5 * z * z) * \
              (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
        
        # Adjust for negative z
        if z < 0:
            cdf = 1 - cdf
        
        return cdf
