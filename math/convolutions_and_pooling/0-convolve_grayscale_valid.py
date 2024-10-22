#!/usr/bin/env python3
"""Module that contains a function for performing valid convolutions on
grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.
    
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
        numpy.ndarray: A numpy array containing the convolved images, with
        shape (m, h-kh+1, w-kw+1).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Create an output array to store the results of the convolution
    output = np.zeros((m, output_h, output_w))

    # Perform convolution for all images at once using only two loops
    for i in range(m):
        image = images[i]
        for x in range(output_h):
            image_slice = image[x:x+kh, :]
            output[i, x, :] = np.sum(image_slice[:, None, :] * \
              kernel[:, None, :], axis=(1, 2))
    
    return output
