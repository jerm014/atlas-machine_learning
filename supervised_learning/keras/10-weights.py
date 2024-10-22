#!/usr/bin/env python3
"""Module that provides functions to save and load a model's weights."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights.

    Args:
        network:                     The model whose weights should be saved.
        filename (str):              The path of the file to save weights to.
        save_format (str, optional): Format to save weights, defaults to Keras.

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights.

    Args:
        network:        The model to which the weights should be loaded.
        filename (str): The path of the file to load the weights from.

    Returns:
        None
    """
    network.load_weights(filename)
    return None
