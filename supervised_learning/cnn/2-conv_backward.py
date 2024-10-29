#!/usr/bin/env python3
"""Module for back propagation over convolutional layer"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """documentation here"""

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == 'same':
        ph = ((h_new - 1) * sh + kh - h_prev + 1) // 2
        pw = ((w_new - 1) * sw + kw - w_prev + 1) // 2
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    dA_prev[
                      i,
                      h*sh:h*sh+kh,
                      w*sw:w*sw+kw,
                      :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += dA_prev_pad[
                      i,
                      h*sh:h*sh+kh,
                      w*sw:w*sw+kw,
                      :] * dZ[i, w, h, c]
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev[:, :h_prev, :w_prev, :]

    return dA_prev, dW, db
