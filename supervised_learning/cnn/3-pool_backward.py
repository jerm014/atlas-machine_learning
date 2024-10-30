#!/usr/bin/env python3
"""Module for back propagation over pooling layer"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the output of the pooling layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
            output of the previous layer.
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
            the pooling.
        stride: tuple of (sh, sw) containing the strides for the pooling.
        mode: 'max' or 'avg', indicating whether to perform maximum or average
            pooling, respectively.

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev).
    """

    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape  # c_new should be equal to c?
    kh, kw = kernel_shape
    sh, sw = stride

    # initialize output gradient to zeros
    dA_prev = np.zeros_like(A_prev)

    # transpose dimensions to bring channels to axis 1 for vectorization
    A_prev = A_prev.transpose(0, 3, 1, 2)  # Shape: (m, c, h_prev, w_prev)
    dA_prev = dA_prev.transpose(0, 3, 1, 2)  # Shape: (m, c, h_prev, w_prev)
    dA = dA.transpose(0, 3, 1, 2)  # Shape: (m, c, h_new, w_new)

    for h in range(h_new):
        for w in range(w_new):
            # define slice corners
            vert_start = h * sh
            vert_end = vert_start + kh
            horiz_start = w * sw
            horiz_end = horiz_start + kw

            if mode == 'max':
                # slice A_prev and create mask of max values
                a_slice = A_prev[:, :,
                                 vert_start:vert_end, horiz_start:horiz_end]
                mask = (a_slice == np.max(a_slice, axis=(2, 3), keepdims=True))
                # distribute gradient according to mask
                dA_prev[:, :, vert_start:vert_end, horiz_start:horiz_end] += \
                    mask * dA[:, :, h:h+1, w:w+1]
            elif mode == 'avg':
                # compute gradient for average pooling
                da = dA[:, :, h:h+1, w:w+1] / (kh * kw)
                shape = (m, c, kh, kw)
                # distribute gradient evenly across the poolng window
                dA_prev[:, :, vert_start:vert_end, horiz_start:horiz_end] += \
                    np.ones(shape) * da

    # transpose back
    dA_prev = dA_prev.transpose(0, 2, 3, 1)
    return dA_prev
