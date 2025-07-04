#!/usr/bin/env python3
"""rotate an image"""
import tensorflow as tf


def rotate_image(image):
    """Rotate an image by 90 degrees counter-clockwise"""
    return tf.image.rot90(image, k=1)
