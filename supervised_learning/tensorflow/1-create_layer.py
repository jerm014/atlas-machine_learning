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
    initializer = tf.variance_scaling_initializer(mode="fan_avg")
    weights = tf.Variable(initializer([int(prev.shape[1]), n]))
    biases = tf.Variable(tf.zeros([n]))
    z = tf.matmul(prev, weights) + biases

    if activation is None:
        return z
    else:
        return activation(z)
