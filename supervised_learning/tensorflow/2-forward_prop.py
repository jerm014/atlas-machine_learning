#!/usr/bin/env python3
"""Module containing the forward_prop function"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.Tensor):      The placeholder for the input data.
        layer_sizes (list): A list containing the number of nodes in each layer
                            of the network.
        activations (list): A list containing the activation functions for each
                            layer of the network.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """
    if len(layer_sizes) != len(activations):
        raise ValueError("The number of layer sizes must match the number " + /
                         "of activation functions.")

    prev_layer = x
    for i in range(len(layer_sizes)):
        prev_layer = create_layer(prev_layer, layer_sizes[i], activations[i])

    return prev_layer
