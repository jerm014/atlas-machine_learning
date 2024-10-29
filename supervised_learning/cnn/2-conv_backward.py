#!/usr/bin/env python3
"""Module for back propagation over convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over convolutional layer of neural network

    Args:
        dZ: numpy.ndarray (m, h_new, w_new, c_new) cont. partial derivatives
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) output of prev layer
        W: numpy.ndarray (kh, kw, c_prev, c_new) kernels for convolution
        b: numpy.ndarray (1, 1, 1, c_new) biases
        padding: string, either 'same' or 'valid'
        stride: tuple (sh, sw) cont. strides for convolution
    Returns:
        Partial derivatives with respect to prev layer, kernels, and biases
    """
    m = dZ.shape[0]
    h_new, w_new, c_new = dZ.shape[1], dZ.shape[2], dZ.shape[3]
    h_prev, w_prev, c_prev = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]
    kh, kw = W.shape[0], W.shape[1]
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    else:
        ph = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev, 0) // 2

    pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev_pad = np.pad(A_prev, pad_width, mode='constant')
    dA_prev_pad = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw
                dA_prev_pad[:, h_start:h_end, w_start:w_end, :] += \
                    W[:, :, :, k] * dZ[:, i:i+1, j:j+1, k:k+1]
                slice_A = A_prev_pad[:, h_start:h_end, w_start:w_end, :]
                for m_i in range(m):
                    dW[:, :, :, k] += slice_A[m_i] * dZ[m_i, i, j, k]

    if padding == 'valid':
        dA_prev = dA_prev_pad
    else:
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
