#!/usr/bin/env python3
"""Module that provides functions to save and load a model's configuration in
JSON format."""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Args:
        network:        The model whose configuration should be saved.
        filename (str): The path of the file that the configuration should be
                        saved to.

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration.

    Args:
        filename (str): The path of the file containing the model's
                        configuration in JSON format.

    Returns:
        The loaded model.
    """
    with open(filename, 'r') as f:
        config = f.read()
    model = K.models.model_from_json(config)
    return model
