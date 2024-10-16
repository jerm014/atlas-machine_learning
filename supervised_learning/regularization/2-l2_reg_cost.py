#!/usr/bin/env python3
""" module for 12_reg_cost function for task 2 of project 2297 """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
    cost: tensor containing the cost of the network without L2 regularization
    model: Keras model that includes layers with L2 regularization

    Returns:
    tensor containing the total cost accounting for L2 regularization
    """
    l2_costs = []
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            l2_cost = tf.nn.l2_loss(layer.kernel)
            l2_costs.append(l2_cost)

    return tf.convert_to_tensor(l2_costs)
