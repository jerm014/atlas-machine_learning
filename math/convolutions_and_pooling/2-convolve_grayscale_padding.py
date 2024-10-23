#!/usr/bin/env python3
"""Module for performing convolution on grayscale images with custom
padding."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform convolution on grayscale images with custom padding.

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

      padding : tuple
        Custom padding values (ph, pw) where:
        - ph: padding for height
        - pw: padding for width

    Returns:
      numpy.ndarray
        Convolved images with shape (m, oh, ow) where:
        - m: number of images
        - oh: output height = h + 2*ph - kh + 1
        - ow: output width = w + 2*pw - kw + 1
    """
    # Get dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate output dimensions
    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    # Create padded input array with zeros
    padded_input = np.pad(
      images, 
      ((0, 0), (ph, ph), (pw, pw)),
      mode='constant', 
      constant_values=0
    )

    # Initialize output array
    output = np.zeros((m, oh, ow))

    # Perform convolution using only two for loops
    for i in range(oh):
        for j in range(ow):
            # Extract window and perform element-wise multiplication w kernel
            # Sum across spatial dimensions for each image
            output[:, i, j] = np.sum(
                padded[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return output
