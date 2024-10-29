#!/usr/bin/env python3
"""Module for forward propagation over convolutional layer"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over convolutional layer of neural network
    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) output of prev layer
        W: numpy.ndarray (kh, kw, c_prev, c_new) kernels for convolution
        b: numpy.ndarray (1, 1, 1, c_new) biases
        activation: activation function to be applied
        padding: string, either 'same' or 'valid'
        stride: tuple (sh, sw) containing strides for convolution
    Returns:
        Output of convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
        h_new = (h_prev - kh) // sh + 1
        w_new = (w_prev - kw) // sw + 1
    else:
        ph = max((h_prev * sh - h_prev + kh - sh) // 2, 0)
        pw = max((w_prev * sw - w_prev + kw - sw) // 2, 0)
        h_new = h_prev
        w_new = w_prev

    pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev_pad = np.pad(A_prev, pad_width, mode='constant')

    output = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw
                slice_A = A_prev_pad[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, k] = np.sum(slice_A * W[:, :, :, k],
                                            axis=(1, 2, 3))

    output = output + b
    return activation(output)
