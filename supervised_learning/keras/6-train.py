#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent with
validation data and early stopping."""

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent, with optional early
    stopping.

    Args:
        network:                           The model to train.
        data (numpy.ndarray):              Input data of shape (m, nx).
        labels (numpy.ndarray):            One-hot labels of shape
                                           (m, classes).
        batch_size (int):                  Size of the batch for mini-batch
                                           gradient descent.
        epochs (int):                      Number of passes through data for
                                           mini-batch gradient descent.
        validation_data (tuple, optional): Data to validate the model with, if
                                           not None.
        early_stopping (bool, optional):   Indicates whether early stopping
                                           should be used.
        patience (int, optional):          Patience used for early stopping.
        verbose (bool, optional):          Determines if output should be
                                           printed during training.
        shuffle (bool, optional):          Determines whether to shuffle the
                                           batches every epoch.

    Returns:
        History: The History object generated after training the model.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
