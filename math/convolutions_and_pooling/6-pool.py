#!/usr/bin/env python3
"""Module for performing pooling on images."""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform pooling on images.

    Args:

      images : numpy.ndarray
        Input images with shape (m, h, w, c) where:
        - m: number of images
        - h: height of each image in pixels
        - w: width of each image in pixels
        - c: number of channels

      kernel_shape : tuple
        Shape of the pooling kernel (kh, kw) where:
        - kh: height of the kernel
        - kw: width of the kernel

      stride : tuple
        Stride values (sh, sw) where:
        - sh: stride for height
        - sw: stride for width

      mode : str, optional
        Type of pooling operation:
        - 'max': maximum pooling (default)
        - 'avg': average pooling

    Returns:
      numpy.ndarray
        Pooled images with shape (m, oh, ow, c) where:
        - m: number of images
        - oh: output height = ((h - kh) / sh) + 1
        - ow: output width = ((w - kw) / sw) + 1
        - c: number of channels
    """
    # Get dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    oh = int(((h - kh) / sh) + 1)
    ow = int(((w - kw) / sw) + 1)

    # Initialize output array
    output = np.zeros((m, oh, ow, c))

    for i in range(oh):  # Loop over height
        for j in range(ow):  # ...over width
            # Calculate the window indices
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract window
            window = images[:, h_start:h_end, w_start:w_end, :]

            # Apply pooling operation based on mode
            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(window, axis=(1, 2))
            else:
                raise ValueError("do better. mode must be 'max' or 'avg'")

    return output
