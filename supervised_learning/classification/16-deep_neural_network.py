#!/usr/bin/env python3
"""Module containing the DeepNeuralNetwork class for binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor for the deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for ll in range(1, self.L + 1):
            if not isinstance(layers[ll-1], int) or layers[ll-1] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f'W{ll}'] = np.random.randn(layers[ll-1],
                                                     nx if ll == 1 else
                                                     layers[ll-2]) * \
                np.sqrt(2 / (nx if ll == 1 else layers[ll-2]))
            self.weights[f'b{ll}'] = np.zeros((layers[ll-1], 1))
