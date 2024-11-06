#!/usr/bin/env python3
"""Build an inception block."""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    "Going Deeper with Convolutions" (2014).

    Args:
      A_prev:  output fr_om the previous layer
      filters: tuple/list with F1, F3R, F3, F5R, F5, FPP

    Returns:
      concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 convolution
    conv1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        activation='relu',
        padding='same')(A_prev)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv3x3_reduce = K.layers.Conv2D(
        filters=F3R,
        kernel_size=1,
        activation='relu',
        padding='same')(A_prev)
    conv3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        activation='relu',
        padding='same')(conv3x3_reduce)

    # Branch 3: 1x1 convolutin followed by 5x5 convolution
    conv5x5_reduce = K.layers.Conv2D(
        filters=F5R,
        kernel_size=1,
        activation='relu',
        padding='same')(A_prev)
    conv5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=5,
        activation='relu',
        padding='same')(conv5x5_reduce)

    # Branch 4: 3x3 max pooling followd by 1x1 convolution
    maxpool = K.layers.MaxPooling2D(
        pool_size=3,
        strides=1,
        padding='same')(A_prev)
    conv_after_pool = K.layers.Conv2D(
        filters=FPP,
        kernel_size=1,
        activation='relu',
        padding='same')(maxpool)

    # Concatenate all the branches
    output = K.layers.concatenate(
        [conv1x1, conv3x3, conv5x5, conv_after_pool],
        axis=-1)

    return output
