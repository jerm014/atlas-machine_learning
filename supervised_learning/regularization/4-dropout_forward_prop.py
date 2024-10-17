#!/usr/bin/env python3
""" module for dropout_forward_prop function for task 4 of project 2297 """
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    X : numpy.ndarray of shape (nx, m)
        The input data for the network.
        nx is the number of input faetures
        m is the number of data points
    weights : dict
        The weights and biases of the neural network.
    L : int
        The number of layers in the network.
    keep_prob : float
        The probablity that a node will be kept.

    Returns:
    dict
        A dictionary containing the outputs of each layer and 
        the dropout mask used on each layer.

    Look at me:
        Writing good documentaiton.
    """
    cache = {}
    A = X
    cache['A0'] = A

    for layer in range(1, L + 1):
        W = weights[f'W{l}']
        b = weights[f'b{l}']
        Z = np.dot(W, A) + b

        if layer == L:
            # Last layer uses softmax activation ok?
            cache[f'A{layer}'] = np.exp(z) / np.sum(np.exp(z),
                                                    axis=0,
                                                    keepdims=True)
        else:
            # Other layers use tanh activation
            A = np.tanh(Z)

            # Apply da dropout
            drop = np.random.rand(* A.shape) < keep_prob
            A *= drop
            A /= keep_prob
            D = drop.astype(int)

            # Save the dropout mask
            cache[f'D{l}'] = D

            cache[f'A{l}'] = A

    return cache
