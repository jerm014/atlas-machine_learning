#!/usr/bin/env python3
"""
Task 2: Initialise
Create the class MultiNormal that represents a Multivariate Normal
distribution:

class constructor def __init__(self, data):
data is a numpy.ndarray of shape (d, n) containing the data set:
n is the number of data points
d is the number of dimensions in each data point
If data is not a 2D numpy.ndarray, raise a TypeError with the message data
must be a 2D numpy.ndarray
If n is less than 2, raise a ValueError with the message data must contain
multiple data points
Set the public instance variables:
mean - a numpy.ndarray of shape (d, 1) containing the mean of data
cov - a numpy.ndarray of shape (d, d) containing the covariance matrix data
You are not allowed to use the function numpy.cov

Task 3: PDF
Update the class MultiNormal:

public instance method def pdf(self, x): that calculates the PDF at a data
point:
x is a numpy.ndarray of shape (d, 1) containing the data point whose PDF
should be calculated
d is the number of dimensions of the Multinomial instance
If x is not a numpy.ndarray, raise a TypeError with the message x must be a
numpy.ndarray
If x is not of shape (d, 1), raise a ValueError with the message x must have
the shape ({d}, 1)
Returns the value of the PDF
You are not allowed to use the function numpy.cov


"""
import numpy as np


class MultiNormal:
    """Class representing a Multivarite Normal distribution.

    This class calculates and stores the mean vector and covariance matrix
    for a given dataset of multiple points.

    Attributes:
        mean: numpy.ndarray of shape (d, 1) contianing the mean vector
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
    """
    TE_nottwodarray = "data must be a 2D numpy.ndarray"
    TE_xnotarray = "x must be a numpy.ndarray"
    VE_less2points = "data must contain multiple data points"

    def __init__(self, data: np.ndarray) -> None:
        """Initialize MultiNormal with dataset.

        Args:
            data: numpy.ndarray of shape (d, n) where d is dimensions
                 and n is number of poits

        Raises:
            TypeError: If data is not a 2D numpy.ndarray
            ValueError: If data contains less than 2 points
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError(self.TE_nottwodarray)

        d, n = data.shape
        if n < 2:
            raise ValueError(self.VE_less2points)

        self.mean = np.sum(data, axis=1).reshape(d, 1) / n

        centered_data = data - self.mean

        self.cov = np.dot(centered_data, centered_data.T) / (n - 1)

    def pdf(self, x: np.ndarray) -> float:
        """Calculate the Probability Density Function (PDF) at point x.

        Args:
            x: numpy.ndarray of shape (d, 1) contianing the data point
               for PDF calculation, where d is the number of dimensions

        Returns:
            float: The PDF value at point x

        Raises:
            TypeError: If x is not a numpy.ndarray
            ValueError: If x does not have the shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(TE_xnotarray)

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculate determinant and inverse of covariance matrx
        det = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)

        # Calculate (x - u)^T E^-1 (x - u) <- I can't type these symbols
        diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)

        # Calculate normalization constant
        normalization = 1 / np.sqrt((2 * np.pi) ** d * det)

        # Return the PDF value
        return float(normalization * np.exp(exponent))
