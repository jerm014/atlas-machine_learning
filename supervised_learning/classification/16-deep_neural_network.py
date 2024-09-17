#!/usr/bin/env python3
"""Module containing DeepNeuralNetwork class for binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification."""

    def __init__(self, nx, layers):
        """Initialize a DeepNeuralNetwork instance."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            layer_size = layers[i]
            layer_input_size = nx if i == 0 else layers[i-1]
            self.weights['W' + str(i+1)] = (
                np.random.randn(layer_size, layer_input_size) *
                np.sqrt(2 / layer_input_size)
            )
            self.weights['b' + str(i+1)] = np.zeros((layer_size, 1))
