#!/usr/bin/env python3
""" module for l2_reg_create_layer function for task 3 of project 2297 """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    documentation goes here
    """
    # Initialize the weights using VarianceScaling
    kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                               mode='fan_avg')
    # use the keras l2 regularizer
    kernel_regularizer = tf.keras.regularizers.l2(lambtha)
    return tf.keras.layers.Dense(n,
                                 activation=activation,
                                 kernel_regularizer=kernel_regularizer,
                                 kernel_initializer=kernel_initializer)(prev)
