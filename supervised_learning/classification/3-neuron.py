#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """A class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the Neuron.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
                nx is the number of input features to the neuron.
                m is the number of examples.

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels for the input data.
                Shape is (1, m) where m is the number of examples.
            A (numpy.ndarray): Activated output of the neuron for each example.
                Shape is (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
