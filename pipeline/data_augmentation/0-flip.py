#!/usr/bin/env python3

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally
    image: the image
    returns: flipped image
    """
    return tf.image.flip_left_right(image)
