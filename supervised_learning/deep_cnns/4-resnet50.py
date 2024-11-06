#!/usr/bin/env python3
"""Builds a ResNet-50 architecture"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architeture as described in
    "Deep Residual Learning for Image Recognition" (2015).

    Returns:
     - resnet model
    """

    he_normal = K.initializers.he_normal(seed=0)

    # Input layer
    inputs = K.Input(shape=(224, 224, 3), name='input_1')

    # Initial convolutional layer
    X = K.layers.Conv2D(
        64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=he_normal)(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu', name='re_lu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same')(X)

    # Stage 1
    X = projection_block(X, filters=[64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, filters=[64, 64, 256])

    # Stage 2
    X = projection_block(X, filters=[128, 128, 512], s=2)
    for _ in range(3):
        X = identity_block(X, filters=[128, 128, 512])

    # Stage 3
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    for _ in range(5):
        X = identity_block(X, filters=[256, 256, 1024])

    # Stage 4
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    for _ in range(2):
        X = identity_block(X, filters=[512, 512, 2048])

    # Avg Pooling
    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)

    # Flaten the thing
    X = K.layers.Flatten()(X)

    # Output Layer
    outputs = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=he_normal)(X)

    return K.Model(inputs=inputs, outputs=outputs)
