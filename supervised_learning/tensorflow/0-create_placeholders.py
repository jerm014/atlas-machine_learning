#!/usr/bin/env python3
"""create_placeholders function for project 2287 task 0"""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network.

    Args:
        nx (int):      The number of feature columns in our data.
        classes (int): The number of classes in our classifier.

    Returns:
        tuple: Two tf.placeholder objects named x and y, respectively.
            x is the placeholder for the input data to the neural network.
            y is the placeholder for the one-hot labels for the input data.
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
