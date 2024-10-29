#!/usr/bin/env python3
"""Module for back propagation over convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """redo documentation, jermy."""

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_new) // 2
        pw = ((w_prev - 1) * sw + kw - w_new) // 2
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant',
        constant_values=0)
    dA_prev_pad = np.pad(
        dA_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant',
        constant_values=0)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                a_slice = a_prev_pad[
                    vert_start:vert_end, horiz_start:horiz_end, :]

                for c in range(c_new):
                    da_prev_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end, :] += \
                        W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if ph == 0 and pw == 0:
            dA_prev[i, :, :, :] = da_prev_pad
        else:
            dA_prev[i, :, :, :] = da_prev_pad[
                ph:-ph or None, pw:-pw or None, :]

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    return dA_prev, dW, db
