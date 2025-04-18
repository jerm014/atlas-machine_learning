#!/usr/bin/env python3
""" task 2 """
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    class for machine translation task using the TED talk
    portuguese to English dataset
    """

    def __init__(self, batch_size, max_len):
        """
        initialize the dataset with training and validation splits and
        tokenizers
        """
        def filter_max_len(pt, en):
            return tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len
            )

        raw_train, raw_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            raw_train
        )

        self.batch_size = batch_size
        self.max_len = max_len

        self.vocab_size_pt = self.tokenizer_pt.vocab_size
        self.vocab_size_en = self.tokenizer_en.vocab_size

        # set these before calling tf_encode which calls encode:
        self.start_token_pt = self.vocab_size_pt
        self.end_token_pt = self.vocab_size_pt + 1
        self.start_token_en = self.vocab_size_en
        self.end_token_en = self.vocab_size_en + 1

        # this should be in a function... we do almost the same thing
        # to train and valid.
        self.data_train = raw_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = raw_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        create sub-word tokenizers for the dataset

        args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                 pt is the portuguese sentence
                 en is the english sentence
        returns:
            tokenizer_pt: portuguese BERT tokenizer
            tokenizer_en: english BERT tokenizer
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes portuguese and english sentences into token sequences

        args:
            pt tf.tensor containing portuguese sentence
            en tf.tensor containing english sentence

        returns:
            tuple of two np.ndarrays
            portuguese tokens with start and end tokens
            english tokens with start and end tokens
        """
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False
        )

        pt_tokens = [self.start_token_pt] + pt_tokens + [self.end_token_pt]
        en_tokens = [self.start_token_en] + en_tokens + [self.end_token_en]

        return tf.convert_to_tensor(pt_tokens, dtype=tf.int64), \
            tf.convert_to_tensor(en_tokens, dtype=tf.int64)

    def tf_encode(self, pt, en):
        """
        tensorflow wrapper for the encode method to use in tf.data
        pipelines

        ags:
            pt: tf.Tensor, portuguese sentence
            en: tf.Tensor, english sentence

        returns:
            result_pt: tf.Tensor with shape (None,) portuguese
            result_en: tf.Tensor with shape (None,) engrish
        """
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
