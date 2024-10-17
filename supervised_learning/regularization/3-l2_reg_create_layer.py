#!/usr/bin/env python3
""" module for l2_reg_create_layer function for task 3 of project 2297 """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    documentation goes here
    """
    # Initialize the weights using He
    initializer = tf.keras.initializers.he_normal()

    # Create the layer with L2 regulraization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )

    # Apply the layer to the previous layers output
    return layer(prev)
