#!/usr/bin/env python3
"""Module that provides functions to save and load an entire Keras model."""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    Args:
        network:        The model to save.
        filename (str): The path of the file that the model should be saved to.

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model.

    Args:
        filename (str): The path of the file to load the model from.

    Returns:
        The loaded model.
    """
    return K.models.load_model(filename)
