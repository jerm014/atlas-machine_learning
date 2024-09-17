#!/usr/bin/env python3
""" Module containing the DeepNeuralNetwork class for binary classification """

import numpy as np


class DeepNeuralNetwork:
    """ Defines a deep neural network performing binary classification """

    def __init__(self, nx, layers):
        """ Class constructor for the deep neural network """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for ll in range(1, self.__L + 1):
            layer_size = layers[ll-1]
            prev_layer_size = nx if ll == 1 else layers[ll-2]
            self.__weights[f'W{ll}'] = np.random.randn(layer_size,
                                                       prev_layer_size) * \
                np.sqrt(2 / prev_layer_size)
            self.__weights[f'b{ll}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """ Getter for L """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache['A0'] = X
        for ll in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f'W{ll}'],
                          self.__cache[f'A{ll-1}']) + self.__weights[f'b{ll}']
            self.__cache[f'A{l}'] = 1 / (1 + np.exp(-Z))
        return self.__cache[f'A{self.__L}'], self.__cache
