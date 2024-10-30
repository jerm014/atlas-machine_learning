#!/usr/bin/env python3
"""Module for back propagation over pooling layer"""
import numpy as np


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using TensorFlow.
    
    Args:
        x: tf.placeholder of shape (m, 28, 28, 1) containing the input images.
        y: tf.placeholder of shape (m, 10) containing the one-hot labels.
        
    Returns:
        y_pred: tensor for the softmax activated output.
        train_op: training operation using Adam optimization.
        loss: tensor for the loss of the network.
        accuracy: tensor for the accuracy of the network.
    """
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    he_init = tf.keras.initializers.VarianceScaling(scale=2.0)
    
    # FIRST Conv Layer: 6 filters 5x5, same padding, ReLU activation
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=he_init,
        name='conv1'
    )
    
    # FIRST Max Pooling Layer: 2x2 kernel, 2x2 strides
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=2,
        strides=2,
        name='pool1'
    )
    
    # SECOND Conv Layer: 16 filters 5x5, valid padding, ReLU activation
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=he_init,
        name='conv2'
    )
    
    # SECOND Max Pooling Layer: 2x2 kernel, 2x2 strides
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=2,
        name='pool2'
    )
    
    # flatten the output from the second pooling layer
    flat = tf.layers.flatten(pool2)
    
    # FIRST Fully Connected Layer: 120 nodes, ReLU activation
    fc1 = tf.layers.dense(
        inputs=flat,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=he_init,
        name='fc1'
    )
    
    # SECOND Fully Connected Layer: 84 nodes, ReLU activation
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=he_init,
        name='fc2'
    )
    
    # OUTPUT Layer: 10 nodes (classes), w softmax activation
    logits = tf.layers.dense(
        inputs=fc2,
        units=10,
        kernel_initializer=he_init,
        name='logits'
    )
    
    # apply softmax to lgits to get the final predictions
    y_pred = tf.nn.softmax(logits, name='y_pred')
    
    # define the loss function using softmax cross-entropy
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits),
        name='loss'
    )
    
    # define the optimizer (Adam) and the training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    
    # calculate accuracy
    correct_preds = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32),
                              name='accuracy')
    
    return y_pred, train_op, loss, accuracy
