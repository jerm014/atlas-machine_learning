#!/usr/bin/env python3
"""Module containing update_variables_Adam function"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float):        The learning rate.
        beta1 (float):        The weight used for the first moment.
        beta2 (float):        The weight used for the second moment.
        epsilon (float):      A small number to avoid division by zero.
        var (numpy.ndarray):  The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v (numpy.ndarray):    The previous first moment of var.
        s (numpy.ndarray):    The previous second moment of var.
        t (int):              The time step used for bias correction.

    Returns:
        tuple: A tuple containing:
          - numpy.ndarray: The updated variable.
          - numpy.ndarray: The new first moment.
          - numpy.ndarray: The new second moment.
    """
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var_updated, v_new, s_new
