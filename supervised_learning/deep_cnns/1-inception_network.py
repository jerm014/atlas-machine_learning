#!/usr/bin/env python3
"""Build an inception block."""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in
    "Going Deeper with Convolutions" (2014).

    Returns:
      keras model
    """

    # Input layer
    inputs = K.Input(shape=(224, 224, 3))

    # Layer 1: Conv 7x7 with stride 2
    x = K.layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu')(inputs)
    x = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same')(x)

    # Layer 2: Conv 1x1 and Conv 3x3
    x = K.layers.Conv2D(
        64,
        kernel_size=1,
        padding='same',
        activation='relu')(x)
    x = K.layers.Conv2D(
        192,
        kernel_size=3,
        padding='same',
        activation='relu')(x)
    x = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same')(x)

    # Inception Block 1
    x = inception_block(x, [64, 96, 128, 16, 32, 32])

    # Inception Block 2
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Inception Block 3
    x = inception_block(x, [192, 96, 208, 16, 48, 64])

    # Inception Block 4
    x = inception_block(x, [160, 112, 224, 24, 64, 64])

    # Inception Block 5
    x = inception_block(x, [128, 128, 256, 24, 64, 64])

    # Inception Block 6
    x = inception_block(x, [112, 144, 288, 32, 64, 64])

    # Inception Block 7
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Inception Block 8
    x = inception_block(x, [256, 160, 320, 32, 128, 128])

    # Inception Block 9
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # Global Average Pooling
    x = K.layers.AveragePooling2D(pool_size=7, strides=1)(x)
    x = K.layers.Dropout(rate=0.4)(x)

    # Output Layer
    outputs = K.layers.Dense(1000, activation='softmax')(x)

    # Create model
    model = K.Model(inputs=inputs, outputs=outputs)

    return model
