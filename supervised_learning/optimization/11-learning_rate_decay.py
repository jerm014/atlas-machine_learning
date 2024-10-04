#!/usr/bin/env python3
"""Module containing learning_rate_decay function"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Args:
    alpha (float): The original learning rate.
    decay_rate (float): The weight used to determine the decay rate of alpha.
    global_step (int): The number of passes of gradient descent elapsed.
    decay_step (int): Number of passes before alpha is decayed further.

    Returns:
    float: The updated value for alpha.
    """

    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
