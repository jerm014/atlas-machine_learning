#!/usr/bin/env python3
"""Module that contains a function for performing same convolutions on
grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
    
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
        numpy.ndarray: A numpy array containing the convolved images, with the
                       same dimensions as the input images (m, h, w).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for height and width
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Pad the images with 0s (same padding)
    padded_images = np.pad(images,
                           ((0, 0),
                           (pad_h, pad_h),
                           (pad_w, pad_w)),
                           mode='constant')

    # Initialize the output array with the same height and width as the input
    # images
    output = np.zeros((m, h, w))

    # Perform convolution using only two loops (one for images, one for height)
    for i in range(m):
        for x in range(h):
            # Extract the slice of the padded image that corresponds to the
            # current window in height and width
            image_slice = padded_images[i, x:x + kh, :]
            IS = image_slice[:, np.newaxis, :w + kw]
            K = kernel[:, np.newaxis]
            axis = (0,1)
            # Use numpy's sum to compute convolution across the width for the
            # 'same' convolution
            output[i, x, :] = np.sum(IS * K, axis=axis)

    return output
