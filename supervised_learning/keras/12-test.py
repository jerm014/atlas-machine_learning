#!/usr/bin/env python3
"""Module that provides a function to test a neural network model."""

import tensorflow.keras as K


def test_model(network,
               data,
               labels,
               verbose=True):
    """
    Tests a neural network.

    Args:
        network:                  The network model to test.
        data:                     The input data to test the model with.
        labels:                   The correct one-hot labels of data.
        verbose (bool, optional): Determines if output should be printed
                                  during the testing process.

    Returns:
        tuple: The loss and accuracy of the model with the testing data,
               respectively.
    """
    return network.evaluate(data,
                            labels,
                            verbose=verbose)
