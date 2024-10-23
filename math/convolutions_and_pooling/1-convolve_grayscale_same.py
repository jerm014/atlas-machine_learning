#!/usr/bin/env python3
"""Module for performing 'same' convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a 'same' convolution on grayscale images.

    Args:
        images (numpy.ndarray): A numpy array with shape (m, h, w) containing
                                multiple grayscale images.
            - m (int): The number of images.
            - h (int): The height in pixels of the images.
            - w (int): The width in pixels of the images.
        kernel (numpy.ndarray): A numpy array with shape (kh, kw) containing
                                the kernel for the convolution.
            - kh (int): The height of the kernel.
            - kw (int): The width of the kernel.

    Returns:
        numpy.ndarray: A numpy array containing the convolved images with the
                       same height and width as the input.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for height and width
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images with 0s (same padding)
    # Padding format: ((before_m, after_m),
    #                  (before_h, after_h),
    #                  (before_w, after_w))
    padded_images = np.pad(images,
                           ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    # Initialize the output array with the same height and width as the input
    # images
    output = np.zeros((m, h, w))

    # Perform convolution using only two loops (one for height, one for width)
    for hh in range(h):
        for ww in range(w):
            # Extract the slice of the padded image that corresponds to the
            # current window in height.
            image_slices = padded_images[:, hh:hh+kh, ww:ww+kw]

            # Perform element-wise multiplication between the kernel and each
            # slice, then sum.
            convolve = np.sum(image_slices * kernel, axis=(1, 2))

            # Assign the convolved values to the output
            output[:, hh, ww] = convolve

    return output
