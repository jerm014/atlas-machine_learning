#!/usr/bin/env python3
"""Module containing the create_layer function"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Args:
        prev (tf.Tensor):      The tensor output of the previous layer.
        n (int):               The number of nodes in the layer to create.
        activation (callable): The activation function for this layer.

    Returns:
        tf.Tensor: The tensor output of the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    return layer(prev)
