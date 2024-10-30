#!/usr/bin/env python3
"""Module for back propagation over pooling layer"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs backpropagation over a pooling layer of a neural network.

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
    _, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # init the output gradient to zeros
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    # find corners of the current slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        # use mask to identify the max entry
                        a_slice = a_prev[vert_start:vert_end,
                                         horiz_start:horiz_end, ch]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end, ch] += \
                                    mask * dA[i, h, w, ch]
                    elif mode == 'avg':
                        # distribute the gradient
                        da = dA[i, h, w, ch]
                        shape = (kh, kw)
                        average = da / (kh * kw)
                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                ch] += np.ones(shape) * average
    return
