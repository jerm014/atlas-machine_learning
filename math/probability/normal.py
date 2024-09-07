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
        Calculate the Cumulative Distribution Function (CDF) for a given
        x-value.

        Args:
            x (float): The x-value to calculate the CDF for.

        Returns:
            float: The CDF value for the given x.
        """
        # Get the z-score using the provided function
        z = self.z_score(x)

        # Use numerical integration (Simpson's rule) to approximate the CDF
        num_steps = 10000
        step_size = z / num_steps
        integral = 0

        for i in range(num_steps):
            x0 = i * step_size
            x1 = (i + 1) * step_size
            integral += (self.pdf(x0) + 4 * self.pdf((x0 + x1) / 2) +
                         self.pdf(x1)) * step_size / 6

        return 0.5 + integral
