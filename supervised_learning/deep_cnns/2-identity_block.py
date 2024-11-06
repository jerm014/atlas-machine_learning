#!/usr/bin/env python3
"""Builds an identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    "Deep Residual Learning for Image Recognition" (2015).
    
    Parameters:
    - A_prev: output from the previous layer
    - filters: tuple/list with F11, F3, F12
    
    Returns:
    - The activated output of the identity block
    """

    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal(seed=0)
    X_shortcut = A_prev

    # First compnoent of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', kernel_initializer=he_normal)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third compnoent of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut to main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
