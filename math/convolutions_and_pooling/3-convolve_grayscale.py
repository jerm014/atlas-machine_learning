#!/usr/bin/env python3
"""Module for performing strided convolution on grayscale images."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform convolution operation on grayscale images using a given kernel.

    Args:
      images : numpy.ndarray
        Input grayscale images with shape (m, h, w) where:
        - m: number of images
        - h: height of each image in pixels
        - w: width of each image in pixels

      kernel : numpy.ndarray
        Convolution kernel with shape (kh, kw) where:
        - kh: height of the kernel
        - kw: width of the kernel

      padding : str or tuple, optional
        Padding strategy for the convolution. Can be:
        - 'same': output has same dimensions as input (default)
        - 'valid': no padding
        - tuple of (ph, pw): explicit padding values where:
            ph: padding for height
            pw: padding for width

      stride : tuple, optional
        Stride of the convolution (sh, sw) where:
        - sh: stride for height
        - sw: stride for width
        Default is (1, 1)

    Returns:
    numpy.ndarray
        Convolved images with shape (m, oh, ow) where:
        - m: number of images
        - oh: output height = ((h + 2*ph - kh) / sh) + 1
        - ow: output width = ((w + 2*pw - kw) / sw) + 1
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Handle different padding types
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Calculate output dimensions
    oh = int((h + 2 * ph - kh) / sh) + 1
    ow = int((w + 2 * pw - kw) / sw) + 1

    # Create padded input and output arrays
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant',
                    constant_values=0)
    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            # Calculate the window indices
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract window and perform element-wise multiplication w kernel
            # Sum across dimensions for each image
            output[:, i, j] = np.sum(
                padded[:, h_start:h_end, w_start:w_end] * kernel,
                axis=(1, 2)
            )

    return output
