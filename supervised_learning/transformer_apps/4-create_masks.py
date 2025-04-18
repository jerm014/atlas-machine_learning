#!/usr/bin/env python3
"""task the fourth"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    creates all masks for training/validation:

    args:
          inputs is a tf.Tensor of shape (batch_size, seq_len_in)
            that contains the input sentence
          target is a tf.Tensor of shape (batch_size, seq_len_out)
            that contains the target sentence

    this function should only use tensorflow operations in order to properly
    function in the training step

    returns: encoder_mask, combined_mask, decoder_mask
          encoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) to be applied in the encoder
          combined_mask is the tf.Tensor of shape
            (batch_size, 1, seq_len_out, seq_len_out) used in the 1st
            attention block in the decoder to pad and mask future tokens
            in the input received by the decoder
            
            it takes the maximum between a lookaheadmask and the decoder
            target padding mask
          decoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) used in the 2nd attention
            block in the decoder
    """
    encoder_mask = pd_mask(inputs)
    combined_mask = tf.maximum(pd_mask(target), la_mask(tf.shape(target)[1]))
    decoder_mask = pd_mask(inputs)
    return encoder_mask, combined_mask, decoder_mask

def pd_mask(seq):
    """
    creates a padding mask for a given input sequence.

    this mask is used in sequence models to mask out padding 
    tokens (typically 0s) in the input sequence. the output 
    shape is compatible with attention mechanisms in models 
    like transformers.

    args:
        seq (tf.Tensor): a 2d tensor of shape 
        (batch_size, seq_len) containing input sequences with 
        padding tokens as 0.

    returns:
        tf.Tensor: a 4d tensor of shape 
        (batch_size, 1, 1, seq_len), where padding positions 
        are marked with 1.0 and non-padding positions with 0.0.
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def la_mask(siz):
    """
    creates a look-ahead mask to mask future tokens.

    used in autoregressive models to ensure that position i 
    can only attend to positions less than or equal to i.

    args:
        siz (int): the length of the target sequence.

    returns:
        tf.Tensor: a 2d tensor of shape (siz, siz) with 0s in 
        the lower triangle (including diagonal) and 1s 
        elsewhere, preventing attention to future positions.
    """
    return 1 - tf.linalg.band_part(tf.ones((siz, siz)), -1, 0)
