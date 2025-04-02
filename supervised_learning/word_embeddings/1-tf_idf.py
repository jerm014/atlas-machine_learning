#!/usr/bin/env python3
""" Term Frequency-Inverse Document Frequency """
import numpy as np
from collections import Counter
import re
import math


def tf_idf(sentences, vocab=None):
    """ Create a TF-IDF embedding matrix """
    # preprocess to get words
    words_in_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r'\'s\b|\'\b', '', sentence.lower())
        clean_sentence = re.sub(r'[^\w\s]', '', clean_sentence)
        words = clean_sentence.split()
        words_in_sentences.append(words)

    # create vocabulary if not provided
    if vocab is None:
        # flatten the list of words from all sentences
        all_words = [word for words in words_in_sentences for word in words]
        # unique words, sort for consistency
        vocab = sorted(set(all_words))
    else:
        # vocab is provided so clean it too to be consistent
        cleaned_vocab = []
        for word in vocab:
            # remove possessives
            word = re.sub(r'\'s\b|\'\b', '', word.lower())
            # Remove other punctuation
            word = re.sub(r'[^\w\s]', '', word)
            if word:  # only add non-empty strings
                cleaned_vocab.append(word)
        # Remove duplicates that might result from cleaning
        vocab = sorted(set(cleaned_vocab))

    # convert features to a numpy array
    features = np.array(vocab)

    # calculate document frequency for each word
    document_frequency = {}
    for word in features:
        document_frequency[word] = sum(1 for sentence_words in words_in_sentences if word in sentence_words)

    # total number of documents (sentences)
    num_documents = len(sentences)

    # init embedding matrix with zeros (floats for TF-IDF values)
    embeddings = np.zeros((len(sentences), len(features)), dtype=np.float64)

    # fill embedding matrix with TF-IDF values
    for i, words in enumerate(words_in_sentences):
        # count occurrences of each word in the sentence (term frequency)
        word_counts = Counter(words)
        total_words = len(words)

        # calculate TF-IDF for each word in the vocabulary
        for j, word in enumerate(features):
            if word in word_counts:
                # term frequency (TF): count / total words in document
                tf = word_counts[word] / total_words

                # inverse document frequency (IDF):
                # log(total docs / docs containing word)
                idf = math.log(num_documents / document_frequency[word])

                # TF-IDF
                embeddings[i, j] = tf * idf

    return embeddings, features
