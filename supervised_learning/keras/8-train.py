#!/usr/bin/env python3
"""Module that trains a model using mini-batch gradient descent with validation
data, early stopping, learning rate decay, and model checkpointing."""

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent, with optional early
    stopping, learning rate decay, and model checkpointing.

    Args:
        network:                              The model to train.
        data (numpy.ndarray):                 Input data of shape (m, nx).
        labels (numpy.ndarray):               One-hot labels of shape
                                              (m, classes).
        batch_size (int):                     Size of the batch for mini-batch
                                              gradient descent.
        epochs (int):                         Number of passes through data for
                                              mini-batch gradient descent.
        validation_data (tuple, optional):    Data to validate the model with,
                                              if not None.
        early_stopping (bool, optional):      Indicates whether early stopping
                                              should be used.
        patience (int, optional):             Patience used for early stopping.
        learning_rate_decay (bool, optional): Indicates whether learning rate
                                              decay should be used.
        alpha (float, optional):              Initial learning rate.
        decay_rate (float, optional):         Decay rate for inverse time decay
                                              of learning rate.
        save_best (bool, optional):           Indicates whether to save the
                                              best model.
        filepath (str, optional):             File path where the model should
                                              be saved.
        verbose (bool, optional):             Determines if output should be
                                              printed during training.
        shuffle (bool, optional):             Determines whether to shuffle the
                                              batches every epoch.

    Returns:
        History: The History object generated after training the model.
    """
    def scheduler(epoch):
        """Inverse time decay learning rate scheduler."""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data is not None:
        lr_schedule = K.callbacks.LearningRateScheduler(scheduler,
                                                        verbose=1)
        callbacks.append(lr_schedule)

    if save_best and validation_data is not None:
        checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                 monitor='val_loss',
                                                 save_best_only=True)
        callbacks.append(checkpoint)

    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
