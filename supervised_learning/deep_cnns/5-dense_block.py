#!/usr/bin/env python3
"""Build a dense block."""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a Dense Block(TM) as described in
    "Densely Connected Convolutional Networks".

    Args:
      X:           output fr_om the previous layer
      nb_filters:  integer representing the number of filters in X
      growth_rate: growth rate for the Dense Block(TM)
      layers:      number of layers in the Dense Block(TM)

    Returns:
      The concatenated output of each layer within the Dense Block(TM)
      The number of filters within the concatted outputs
    """

    he_normal = K.initializers.he_normal(seed=0)

    # Loop over the number of layers
    for i in range(layers):
        # Bottleneck layer
        # Batch Normalization->ReLU->1x1 Convloution (4 * growth_rate filters)
        X1 = K.layers.BatchNormalization(axis=3)(X)
        X1 = K.layers.Activation('relu')(X1)
        X1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=he_normal)(X1)

        # Batch Normaliaztion -> ReLU -> 3x3 Convolution (growth_rate filters)
        X1 = K.layers.BatchNormalization(axis=3)(X1)
        X1 = K.layers.Activation('relu')(X1)
        X1 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=he_normal)(X1)

        # Concatenate the input (X) with output (X1) along the channel's axis
        X = K.layers.Concatenate(axis=3)([X, X1])

        # Update the number of filters
        nb_filters += growth_rate

    return X, nb_filters
