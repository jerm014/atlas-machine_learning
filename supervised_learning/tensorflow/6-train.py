#!/usr/bin/env python3
"""Module containing the train function"""

import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier. Fancy.

    Args:
        X_train (numpy.ndarray): The training input data.
        Y_train (numpy.ndarray): The training labels.
        X_valid (numpy.ndarray): The validation input data.
        Y_valid (numpy.ndarray): The validation labels.
        layer_sizes (list):      The number of nodes in each layer.
        activations (list):      The activation functions for each layer.
        alpha (float):           The learning rate.
        iterations (int):        The number of iterations to train over.
        save_path (str):         Where to save the model.

    Returns:
        str: The spot where the model was written out. Booyah.
    """
    tf.reset_default_graph()

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            t_cost, t_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: Y_train})
            v_cost, v_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {t_cost}")
                print(f"\tTraining Accuracy: {t_accuracy}")
                print(f"\tValidation Cost: {v_cost}")
                print(f"\tValidation Accuracy: {v_accuracy}")

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
