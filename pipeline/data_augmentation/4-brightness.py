#!/usr/bin/env python3
"""change brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly change the brightness of an image"""
    return tf.image.random_brightness(image, max_delta)
