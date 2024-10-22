#!/usr/bin/env python3
""" module to build a keras model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras Functional API without using Sequential
    or Input classes.

    Args:
        nx (int):           Number of input features to the network.
        layers (list):      List containing the number of nodes in each layer.
        activations (list): List containing the activation functions for each
                            layer.
        lambtha (float):    L2 regularization parameter.
        keep_prob (float):  Probability that a node will be kept during
                            dropout.

    Returns:
        Keras.Model: The constructed Keras model.
    """

    # Create an InputLayer and get its input tensor
    input_layer = K.layers.InputLayer(input_shape=(nx,))
    x = input_layer.input  # This is the input tensor

    # Build the rest of the model
    y = x
    for i in range(len(layers)):
        # Add Dense layer
        y = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(y)
        # Add Dropout layer after each layer except the last
        if i < len(layers) - 1:
            y = K.layers.Dropout(rate=1 - keep_prob)(y)

    # Create the model
    model = K.models.Model(inputs=x, outputs=y)
    return model
