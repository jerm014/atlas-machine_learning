#!/usr/bin/env python3
"""Module for back propagation over convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    Parameters:
        dZ:      numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
                 partial derivatives with respect to the unactivated output of
                 the convolutional layer
                  - m: integer of examples
                  - h_new: integer height of the output
                  - w_new: integer width of the output
                  - c_new: integer channels in the output
        A_prev:  numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                 the output of the previous layer
                  - m: integer of examples
                  - h_prev: integer height of the previous layer
                  - w_prev: integer width of the previous layer
                  - c_prev: integer channels in the previous layer
        W:       numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
                 kernels for the convolution
                  - kh: integer filter height
                  - kw: integer filter width
                  - c_prev: integer channels in the previous layer
                  - c_new: integer channels in the output
        b:       numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
                 applied to the convolution
        padding: string 'same' or 'valid', indicating the type of padding used
        stride:  tuple of (sh, sw) containing the strides for the convolution
                  - sh: integer stride height
                  - sw: integer stride width

    Returns:
        dA_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                 the partial derivatives with respect to the previous layer
        dW:      numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
                 kernels for the convolution
        db:      numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
                 applied to the convolution
    """

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + \
            ((h_prev - 1) * sh + kh - h_prev) % 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + \
            ((w_prev - 1) * sw + kw - w_prev) % 2
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev, ((0,0), (ph, ph), (pw, pw), (0,0)), mode='constant')
    dA_prev_padded = np.pad(
        dA_prev, ((0,0), (ph, ph), (pw, pw), (0,0)), mode='constant')

    for h in range(h_new):
        for w in range(w_new):

            vert_start = h * sh
            vert_end = vert_start + kh
            horiz_start = w * sw
            horiz_end = horiz_start + kw

            # Slice A_prev_padded to get the current slice
            a_slice = A_prev_padded[
                :, vert_start:vert_end, horiz_start:horiz_end, :]

            # Reshape dZ to align dimensions for broadcasting
            dZ_slice = dZ[:,
                h, w, :][:, np.newaxis, np.newaxis, np.newaxis, :]

            # Update gradients for the input
            dA_prev_padded[
                :, vert_start:vert_end, horiz_start:horiz_end, :] += np.sum(
                W[np.newaxis, :, :, :, :] * dZ_slice, axis=4)

            # Update gradients for the kernels
            dW += np.sum(a_slice[:, :, :, :, np.newaxis] * dZ_slice, axis=0)

    if ph == 0 and pw == 0:
        dA_prev = dA_prev_padded
    else:
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
