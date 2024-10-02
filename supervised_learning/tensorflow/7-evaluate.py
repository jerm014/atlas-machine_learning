#!/usr/bin/env python3
"""Module containing the evaluate function"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluate the output of a neural network using tensorflow, oh yeah

    Args:
        X (numpy.ndarray): The input data to evaluate
        Y (numpy.ndarray): The one-hot labels for X
        save_path (str):   The location to load the model from

    Returns:
        tuple: The network's (prediction, accuracy, loss)
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        prediction, acc, cost = sess.run([y_pred, accuracy, loss], 
                                         feed_dict={x: X, y: Y})

    return prediction, acc, cost
