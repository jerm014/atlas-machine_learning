#!/usr/bin/env python3
""" module to build a keras model """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras.

    Args:
        nx (int):           Number of input features to the network.
        layers (list):      List containing the number of nodes in each layer.
        activations (list): List containing the activation functions for each
                            layer.
        lambtha (float):    L2 regularization parameter.
        keep_prob (float):  Probability that a node will be kept during dropout.

    Returns:
        keras.Model: The constructed Keras model.
    """

    model = K.models.Sequential()
    for i in range(len(layers)):
        if i == 0:
            # Add the first layer with input shape
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            # Add subsequent layers
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))
        if i < len(layers) - 1:
            # Add dropout after each layer except the last
            model.add(K.layers.Dropout(rate=1 - keep_prob))
    return model
