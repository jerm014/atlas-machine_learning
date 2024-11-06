#!/usr/bin/env python3
"""Builds a projection block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    "Deep Residual Learning for Image Recognition" (2015).

    Args:
    - A_prev: output fr_om the previous layer
    - filters: tuple/list with F11, F3, F12
    - s: stride for the first convolution in main path and shortcut

    Returns:
    - The activated output of the projection block
    """

    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal(seed=0)
    X_shortcut = A_prev

    ##### MAIN PATH #####
    # First component
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=he_normal)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation(' relu')(X)

    # Second component
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=he_normal)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    # No activtion here

    ##### SHORTCUT PATH #####
    X_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=he_normal)(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)
    # No activtion here

    # Add shortcut to main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
