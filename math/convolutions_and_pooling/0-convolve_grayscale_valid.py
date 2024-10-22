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

    # Initialize the output array
    output = np.zeros((m, output_h, output_w))

    # Loop over each image
    for i in range(m):
        # Loop over the height of the output
        for x in range(output_h):
            # Extract the slice of the image that corresponds to the current window in height
            image_slice = images[i, x:x + kh, :]
            # Use numpy's stride tricks to create a sliding window over the width
            # This will avoid the third loop
            sub_matrices = np.lib.stride_tricks.as_strided(
                image_slice,
                shape=(output_w, kh, kw),
                strides=(image_slice.strides[1],
                image_slice.strides[0],
                image_slice.strides[1])
            )
            # Perform element-wise multiplication and sum over kh and kw dimensions
            output[i, x, :] = np.tensordot(sub_matrices,
                                           kernel,
                                           axes=([1, 2], [0, 1]))
    
    return output
