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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            layer_size = layers[i]
            input_size = nx if i == 0 else layers[i - 1]
            self.__weights[f'W{i+1}'] = np.random.randn(
                layer_size, input_size) * np.sqrt(2 / input_size)
            self.__weights[f'b{i+1}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            Z = (np.matmul(self.__weights[f'W{i}'], self.__cache[f'A{i-1}']) +
                 self.__weights[f'b{i}'])
            self.__cache[f'A{i}'] = 1 / (1 + np.exp(-Z))
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression."""
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        dZ = cache[f'A{self.__L}'] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache[f'A{i-1}']
            dW = (1/m) * np.matmul(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.matmul(self.__weights[f'W{i}'].T, dZ)
            dZ = dA_prev * (A_prev * (1 - A_prev))
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db
