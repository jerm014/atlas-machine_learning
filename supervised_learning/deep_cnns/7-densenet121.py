#!/usr/bin/env python3
"""builds the DenseNet-121 architecture with function densenet121"""
from tensorflow import keras as K


def densenet121(growth_rate=32, compression=1.0):
    """
    Build the DenseNet-121 architecture

    Args:
        growth_rate: growth rate
        compression: compression factor
    Returns:
        keras model
    """
    # Define the input layer
    input_layer = K.layers.Input(shape=(224, 224, 3))

    # Define the initial convolutional layer
    init_conv = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer='he_normal'
    )(input_layer)
    init_conv = K.layers.BatchNormalization()(init_conv)
    init_conv = K.layers.Activation('relu')(init_conv)

    # Define the initial pooling layer
    init_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(init_conv)

    # Define the dense block and transition layer
    dense_block, nb_filters = dense_block(init_pool, 2 * growth_rate,
                                          growth_rate)
    transition_layer = transition_layer(dense_block, nb_filters, compression)

    # Define the dense block and transition layer
    dense_block, nb_filters = dense_block(transition_layer, nb_filters, 
                                          rowth_rate)
    transition_layer = transition_layer(dense_block, nb_filters, compression)

    # Define the dense block and transition layer
    dense_block, nb_filters = dense_block(transition_layer, nb_filters,
                                          growth_rate)
    transition_layer = transition_layer(dense_block, nb_filters, compression)

    # Define the dense block
    dense_block, nb_filters = dense_block(transition_layer, nb_filters,
                                          growth_rate)

    # Define the global average pooling layer
    global_avg_pool = K.layers.GlobalAveragePooling2D()(dense_block)

    # Define the output layer
    output_layer = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer='he_normal'
    )(global_avg_pool)

    # Create the model
    model = K.models.Model(inputs=input_layer, outputs=output_layer)

    return model
