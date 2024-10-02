#!/usr/bin/env python3
"""Module containing the calculate_loss function"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.
      y is a placeholder for the labels of the input data
      y_pred is a tensor containing the network’s predictions
      Returns: a tensor containing the loss of the prediction
    """
    cewl = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    loss = tf.reduce_mean(cewl)
    return loss
