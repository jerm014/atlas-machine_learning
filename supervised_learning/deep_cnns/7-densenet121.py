#!/usr/bin/env python3
"""builds the DenseNet-121 architecture with function densenet121"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    "Densely Connected Convolutional Networks".

    Parameters:
      growth_rate: growth rate
      compression: compression factor

    Returns:
      keras model
    """

    he_normal = K.initializers.he_normal(seed=0)

    # Input layer
    inputs = K.Input(shape=(224, 224, 3))

    # Initial Batch Normalization and ReLU activation
    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)

    # Initial convolution
    nb_filters = 2 * growth_rate  # Typically 64 filters for DenseNet-121
    X = K.layers.Conv2D(
        nb_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=he_normal)(X)

    # Max pooling
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(
        X,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=6)

    # Transition Layer 1
    X, nb_filters = transition_layer(
        X,
        nb_filters=nb_filters,
        compression=compression)

    # Dense Block 2
    X, nb_filters = dense_block(
        X,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=12)

    # Transition Layer 2
    X, nb_filters = transition_layer(
        X,
        nb_filters=nb_filters,
        compression=compression)

    # Dense Block 3
    X, nb_filters = dense_block(
        X,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=24)

    # Transition Layer 3
    X, nb_filters = transition_layer(
        X,
        nb_filters=nb_filters,
        compression=compression)

    # Dense Block 4
    X, nb_filters = dense_block(
        X,
        nb_filters=nb_filters,
        growth_rate=growth_rate,
        layers=16)

    # Avg Pool
    X = K.layers.AveragePooling2D(pool_size=(7, 7))(X)

    # Output layer
    outputs = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=he_normal)(X)

    return K.Model(inputs=inputs, outputs=outputs)
