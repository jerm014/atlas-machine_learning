#!/usr/bin/env python3
""" Poisson functions """


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution.

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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate the value of the PMF for a given number of "successes".

        Args:
            k (int): The number of "successes".

        Returns:
            float: The PMF value for k.
        """
        k = int(k)
        if k < 0:
            return 0

        # Calculate e^(-λ) using e = 2.7182818285 from the project requirements
        e = 2.7182818285
        e_lambda = e ** (-self.lambtha)

        # Calculate λ^k without math
        lambda_k = self.lambtha ** k

        # Calculate k! without math
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        return (e_lambda * lambda_k) / factorial_k
