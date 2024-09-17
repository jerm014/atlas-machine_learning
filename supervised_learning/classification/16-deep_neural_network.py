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
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer_idx in range(1, self.L + 1):
            layer_input_size = nx if layer_idx == 1 else layers[layer_idx-2]
            layer_size = layers[layer_idx-1]
            
            self.weights['W' + str(layer_idx)] = (
                np.random.randn(layer_size, layer_input_size) * 
                np.sqrt(2 / layer_input_size)
            )
            self.weights['b' + str(layer_idx)] = np.zeros((layer_size, 1))
