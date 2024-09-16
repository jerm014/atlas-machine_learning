#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Class constructor for the Neuron

        Args:
            nx (int): The number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
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
        """Getter method for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter method for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter method for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples

        Returns:
            The private attribute __A
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))  # Sigmoid activation function
        return self.__A
