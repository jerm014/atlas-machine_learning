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

    # Calculate the padding for height and width
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Pad the images with zeros
    padded_images = np.pad(images,
                           ((0, 0),
                           (pad_h, pad_h),
                           (pad_w, pad_w)),
                           mode='constant')

    # Create an output array to store the results of the convolution
    output = np.zeros((m, h, w))

    # Perform convolution for all images at once using only two loops
    for i in range(m):
        image = padded_images[i]
        for x in range(h):
            image_slice = image[x:x+kh, :]
            output[i, x, :] = np.sum(image_slice[:, None, :] * \
              kernel[:, None, :], axis=(1, 2))

    return output
