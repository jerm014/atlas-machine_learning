#!/usr/bin/env python3
"""Module for forward propagation over pooling layer"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over pooling layer of neural network
    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) output of prev layer
        kernel_shape: tuple (kh, kw) containing size of pooling kernel
        stride: tuple (sh, sw) containing strides for pooling
        mode: string, either 'max' or 'avg'
    Returns:
        Output of pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            slice_A = A_prev[:, h_start:h_end, w_start:w_end, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return output
