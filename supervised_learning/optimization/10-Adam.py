#!/usr/bin/env python3
"""Module containing create_Adam_op function"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.

    Args:
    alpha (float): The learning rate.
    beta1 (float): The weight used for the first moment.
    beta2 (float): The weight used for the second moment.
    epsilon (float): A small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.Optimizer: The Adam optimizer.
    """

    return tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2,
                                    epsilon=epsilon)
