#!/usr/bin/env python3
"""Module containing create_batch_norm_layer function"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Args:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function to be used.

    Returns:
    tf.Tensor: The activated output for the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-7

    mean, variance = tf.nn.moments(z, axes=[0])
    z_norm = tf.nn.batch_normalization(
        z, mean, variance, beta, gamma, epsilon)

    if activation is None:
        return z_norm
    return activation(z_norm)
