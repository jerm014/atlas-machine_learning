#!/usr/bin/env python3
""" task 0 """
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Dataset class for machine translation task using the TED talk
    Portuguese to English dataset.
    """

    def __init__(self):
        """
        Initialize the dataset with training and validation splits and
        tokenizers.
        """
        # Load Portuguese to English dataset
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
 
        # Create tokenizers from the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                 pt is the Portuguese sentence
                 en is the English sentence
        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Collect Portuguese and English sentences
        pt_sentences = []
        en_sentences = []

        # Convert dataset to lists of sentences
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode())
            en_sentences.append(en.numpy().decode())

        # Train tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences,
            vocab_size=2*13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences,
            vocab_size=2**13
        )
    
        tokenizer_pt.model_max_length = 2**13
        tokenizer_en.model_max_length = 2**13
    
        return tokenizer_pt, tokenizer_en
