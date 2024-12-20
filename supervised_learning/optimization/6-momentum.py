#!/usr/bin/env python3
"""Module containing create_momentum_op function"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm
    in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        tf.keras.optimizers.Optimizer: The momentum optimizer.
    """

    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
