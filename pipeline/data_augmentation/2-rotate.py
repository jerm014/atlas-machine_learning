#!/usr/bin/env python3

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise
    image: the image
    returns: the rotated image
    """
    return tf.image.rot90(image, k=1)
