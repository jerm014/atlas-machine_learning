#!/usr/bin/env python3
"""Module containing the forward_prop function"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.Tensor):      input data.
        layer_sizes (list): A list containing the number of nodes in each layer
                            of the network.
        activations (list): A list containing the activation functions for each
                            layer of the network.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """

    prev = x
    print(layer_sizes)
    print(activations)
    for i, (n, activation) in enumerate(zip(layer_sizes, activations)):
        with tf.variable_scope(f'layer{i}'):
            prev = create_layer(prev, n, activation)
    return prev
