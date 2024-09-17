#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class for multiclass
classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ Defines a deep neural network performing multiclass classification """

    def __init__(self, nx, layers, activation='sig'):
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

        for l in range(1, self.__L + 1):
            layer_size = layers[l-1]
            prev_layer_size = nx if l == 1 else layers[l-2]
            self.__weights[f'W{l}'] = np.random.randn(layer_size,
                                                      prev_layer_size) * \
                np.sqrt(2 / prev_layer_size)
            self.__weights[f'b{l}'] = np.zeros((layer_size, 1))


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

    @property
    def activation(self):
        """ Getter for activation """
        return self.__activation

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f'W{l}'], self.__cache[f'A{l-1}']) + \
                self.__weights[f'b{l}']
            if l == self.__L:
                # Softmax activation for the output layer
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache[f'A{l}'] = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Activation for hidden layers
                if self.__activation == 'sig':
                    self.__cache[f'A{l}'] = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    self.__cache[f'A{l}'] = np.tanh(Z)
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using categorical cross-entropy """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A + 1e-8))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.eye(Y.shape[0])[np.argmax(A, axis=0)].T
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = Y.shape[1]
        dZ = cache[f'A{self.__L}'] - Y
        for l in reversed(range(1, self.__L + 1)):
            dW = 1/m * np.matmul(dZ, cache[f'A{l-1}'].T)
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                if self.__activation == 'sig':
                    dZ = np.matmul(self.__weights[f'W{l}'].T, dZ) * \
                        (cache[f'A{l-1}'] * (1 - cache[f'A{l-1}']))
                else:  # tanh
                    dZ = np.matmul(self.__weights[f'W{l}'].T, dZ) * \
                        (1 - np.power(cache[f'A{l-1}'], 2))
            self.__weights[f'W{l}'] -= alpha * dW
            self.__weights[f'b{l}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the deep neural network """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, self.__cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.plot(range(0, iterations + 1, step), costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
