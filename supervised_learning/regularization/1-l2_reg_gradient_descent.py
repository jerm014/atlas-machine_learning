#!/usr/bin/env python3
""" module for l2_reg_gradient_descent for task 1 on project 2297 """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.

    Args:
    Y: numpy.ndarray, shape (classes, m), correct labels
    weights: dict, weights and biases of the neural network
    cache: dict, outputs of each layer of the neural network
    alpha: float, learning rate
    lambtha: float, L2 regularization parameter
    L: int, number of layers in the network

    The function updates the weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache["A" + str(l - 1)]
        this_W = weights["W" + str(l)]

        dW = np.dot(dZ, A_prev.T) / m + (lambtha / this_W) * m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        dZ = (1 - np.square(A_prev)) * np.dot(this_W.T, dZ)

        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db
