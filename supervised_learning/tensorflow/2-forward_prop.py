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

    for i, (n, activation) in enumerate(zip(layer_sizes, activations)):
        x = create_layer(x, n, activation)
    return x