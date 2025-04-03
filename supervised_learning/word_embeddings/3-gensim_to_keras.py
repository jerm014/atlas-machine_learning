#!/usr/bin/env python3
""" Convert gensim word2vec model to Keras Embedding """
import tensorflow as tf


def gensim_to_keras(model):
    """ Converts trained gensim Word2Vec model to Keras Embedding layer """

    input_dim, output_dim = model.wv.vectors.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        weights=[model.wv.vectors],
        trainable=True
    )

    return embedding_layer
