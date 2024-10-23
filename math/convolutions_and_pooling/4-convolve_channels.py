#!/usr/bin/env python3
"""Module for performing convolution on images with channels."""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform convolution on images with multiple channels.

    Args:

      images : numpy.ndarray
        Input images with shape (m, h, w, c) where:
        - m: number of images
        - h: height of each image in pixels
        - w: width of each image in pixels
        - c: number of channels

      kernel : numpy.ndarray
        Convolution kernel with shape (kh, kw, c) where:
        - kh: height of the kernel
        - kw: width of the kernel
        - c: number of channels (same as input)

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
    # Get dimensions
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
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
    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    # Create padded input array
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant',
                    constant_values=0)

    # Initialize output array
    output = np.zeros((m, oh, ow))

    # Perform convolution using only two for loops
    for i in range(oh):
        for j in range(ow):
            # Calculate the window indices
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract window and perform element-wise multiply with kernel
            # Sum across height, width, and channel dimensions for each image
            window = padded[:, h_start:h_end, w_start:w_end, :]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
