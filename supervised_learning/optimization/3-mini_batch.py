#!/usr/bin/env python3
"""Module containing create_mini_batches function"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches for training a neural network."""
    m = X.shape[0]
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:min(i + batch_size, m)]
        Y_batch = Y_shuffled[i:min(i + batch_size, m)]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches
