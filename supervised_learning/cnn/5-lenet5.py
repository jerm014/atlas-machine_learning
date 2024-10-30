#!/usr/bin/env python3
"""Module for back propagation over pooling layer"""
from tensorflow import keras as K


def lenet5(X):
  """documentaiotn"""
    he_init = K.initializers.he_normal(seed=0)

    # FIRST conv layer
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            kernel_initializer=he_init, activation='relu')(X)
    # FIRST Max Pooling layer
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # SECOND conv layer
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            kernel_initializer=he_init,
                            activation='relu')(pool1)
    # SECOND Max Pooling layer
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # flatten the output
    flat = K.layers.Flatten()(pool2)

    # fully connected layer with 120 nodes
    fc1 = K.layers.Dense(units=120, kernel_initializer=he_init,
                         activation='relu')(flat)
    # fully connected layer with 84 nodes
    fc2 = K.layers.Dense(units=84, kernel_initializer=he_init,
                         activation='relu')(fc1)
    # fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(units=10, kernel_initializer=he_init,
                            activation='softmax')(fc2)

    # build model
    model = K.Model(inputs=X, outputs=output)

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
