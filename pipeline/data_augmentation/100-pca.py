#!/usr/bin/env python3
"""PCA color augmentation for an image based on AlexNet"""
import tensorflow as tf

# These are the approximate eigenvalues of the covariance matrix of RGB pixel
# values over ImageNet. They correspond to the variance explained by each
# principal component.
_EIGEN_VALUES = tf.constant([0.2175, 0.0188, 0.0045], dtype=tf.float32)

# These are the corresponding eigenvectors (as columns).
# p_i = _EIGEN_VECTORS_MATRIX[:, i]
_EIGEN_VECTORS_MATRIX = tf.constant([
    [-0.5675, -0.5808, -0.5836],
    [ 0.7192, -0.0045, -0.6948],
    [ 0.4009, -0.8140,  0.4203]
], dtype=tf.float32)


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image

    This implements the method described in Section 5.1 of the AlexNet paper.
    "ImageNet Classification with Deep Convolutional Neural Networks"
    by Krizhevsky, Sutskever, and Hinton (2012)

    Args:
        image:  A 3D tf.Tensor with shape (height, width, 3) representing the
                input image. Pixel values are assumed to be tf.float32 in the
                range [0, 1]
        alphas: A tuple or list of 3 floats. These are the random variables
                (typically drawn from a Gaussian distribution with mean 0 and
                stddev 0.1, once per image) to scale the eigenvalues 

    return:
        A 3D tf.Tensor of the same shape and type as image, representing the
        color-augmented image.
    """
    if not isinstance(alphas, tf.Tensor):
        alphas_tf = tf.constant(alphas, dtype=tf.float32)
    else:
        alphas_tf = tf.cast(alphas, dtype=tf.float32)

    # Calculate the term: alpha_i * lambda_i for each principal component 
    # The formula used is [p1,p2,p3][a1*l1, a2*l2, a3*l3]^T 
    # Here, _EIGEN_VALUES are the lambda_i.
    scaled_eigenvalues = alphas_tf * _EIGEN_VALUES

    # Calculate the change to add to RGB channels: P * (alphas * lambdas) 
    # P is the matrix whose columns are eigenvectors p_i.
    # _EIGEN_VECTORS_MATRIX is P.
    # tf.linalg.matvec(A, x) computes A*x.
    rgb_delta = tf.linalg.matvec(_EIGEN_VECTORS_MATRIX, scaled_eigenvalues)

    # Add the calculated RGB change to each pixel of the image.
    # image has shape (H, W, 3) and rgb_change has shape (3,).
    # TensorFlow's broadcasting will add rgb_change to each pixel's RGB values
    augmented_image = image + rgb_delta

    return augmented_image
