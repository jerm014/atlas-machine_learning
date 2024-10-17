#!/usr/bin/env python3
""" module for dropout_gradient_descent function for task 5 of project 2297 """
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradeint descent.
    
    Args:
      Y : numpy.ndarray of shape (classes, m)
          One-hot array of corect labels
      weights : dict
          Dictionary of weights and biases of the neural network
      cache : dict
          Dictionary of outputs and dropout masks of each layer
      alpha : float
          Learning rate
      keep_prob : float
          Probabiltiy that a node will be kept
      L : int
          Number of layers in the network
    
    Returns:
        None (weights are updated in place)
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y
    
    for layer in reversed(range(1, L + 1)):
        A_prev = cache[f'A{layer-1}']
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']
        
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        if layer > 1:
            dA_prev = np.dot(W.T, dZ)
            dA_prev *= cache[f'D{layer-1}']  # Apply dropout mask
            dA_prev /= keep_prob  # Scale the values
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # Derivative of tanh
        
        # Update weights and biases
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db

    # No return statement as weights are updated in place!!
