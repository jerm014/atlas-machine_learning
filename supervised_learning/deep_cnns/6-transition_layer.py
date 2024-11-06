#!/usr/bin/env python3
"""Build a transition layer."""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    "Densely Connected Convolutional Networks".

    Args:
      X:           output fr_om the previous layer
      nb_filters:  integer representing the number of filters in X
      compression: compression factor for the transition layer

    Returns:
      The output of the transition layer
      The number of filters within the output
    """

    he_normal = K.initializers.he_normal(seed=0)

    # Batch Normalization and ReLU activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Compute the number of filters after compression
    nb_filters = int(nb_filters * compression)

    # 1x1 Convolution with compression
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=he_normal)(X)

    # Average Pooling
    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X, nb_filters
