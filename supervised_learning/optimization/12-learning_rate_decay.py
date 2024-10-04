#!/usr/bin/env python3
"""Module containing learning_rate_decay function"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse
    time decay.

    Args:
    alpha (float): The original learning rate.
    decay_rate (float): The weight used to determine the decay rate of alpha.
    decay_step (int): Number of passes before alpha is decayed further.

    Returns:
    tf.keras.optimizers.schedules.LearningRateSchedule: The learning rate
    decay operation.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
