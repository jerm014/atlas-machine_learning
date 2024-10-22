#!/usr/bin/env python3
"""Module that provides a function to make predictions using a neural
network."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network.

    Args:
        network:                  The network model to make the prediction
                                  with.
        data:                     The input data to make the prediction with.
        verbose (bool, optional): Determines if output should be printed
                                  during the prediction process.

    Returns:
        numpy.ndarray: The prediction for the data.
    """
    return network.predict(data, verbose=verbose)
