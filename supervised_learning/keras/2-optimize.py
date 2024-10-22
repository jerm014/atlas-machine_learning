#!/usr/bin/env python3
""" module to optimize a keras model """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical crossentropy
    loss and accuracy metrics.

    Args:
        network:       The Keras model to optimize.
        alpha (float): The learning rate.
        beta1 (float): The first Adam optimization parameter (exponential decay
                       rate for the first moment estimates).
        beta2 (float): The second Adam optimization parameter (exponential
                       decay rate for the second moment estimates).

    Returns:
        None
    """

    # Create an instance of the Adam optimizer with the specified parameters
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1,
                                  beta_2=beta2)

    # Compile the model with the optimizer, loss function, and metrics
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
