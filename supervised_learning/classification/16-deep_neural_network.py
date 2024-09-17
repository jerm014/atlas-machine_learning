#!/usr/bin/env python3
"""Module containing the DeepNeuralNetwork class for binary classification"""

import numpy as np

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor for the deep neural network

        Args:
            nx (int): The number of input features
            layers (list): A list representing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer or if layers is not a list
            ValueError: If nx is less than 1
            TypeError: If the elements in layers are not all positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_size = layers[l-1]
            prev_layer_size = nx if l == 1 else layers[l-2]
            
            # Initialize weights using He et al. method
            self.weights[f'W{l}'] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.weights[f'b{l}'] = np.zeros((layer_size, 1))
