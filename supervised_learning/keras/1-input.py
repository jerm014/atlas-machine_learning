#!/usr/bin/env python3
""" module to build a keras model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras using the Model Subclassing API.

    Args:
        nx (int):           Number of input features to the network.
        layers (list):      List containing the number of nodes in each layer.
        activations (list): List containing the activation functions for each
                            layer.
        lambtha (float):    L2 regularization parameter.
        keep_prob (float):  Probability that a node will be kept during
                            dropout.

    Returns:
        Keras.Model: The constructed Keras model.
    """

    class CustomModel(K.models.Model):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.layers_list = []
            for i in range(len(layers)):
                if i == 0:
                    # First layer with input dimension
                    dense = K.layers.Dense(
                        units=layers[i],
                        activation=activations[i],
                        kernel_regularizer=K.regularizers.l2(lambtha),
                        input_dim=nx
                    )
                else:
                    # Subsequent layers
                    dense = K.layers.Dense(
                        units=layers[i],
                        activation=activations[i],
                        kernel_regularizer=K.regularizers.l2(lambtha)
                    )
                self.layers_list.append(dense)
                if i < len(layers) - 1:
                    # Add dropout after each layer except the last
                    dropout = K.layers.Dropout(rate=1 - keep_prob)
                    self.layers_list.append(dropout)

        def call(self, inputs):
            x = inputs
            for layer in self.layers_list:
                x = layer(x)
            return x

    model = CustomModel()
    return model
