#!/usr/bin/env python3
"""bag o words"""
import numpy as np
from collections import Counter


def bag_of_words(sentences, vocab=None):
    """ Create a bag of words embedding matrix. """
    # preprocess to get words
    words_in_sentences = []
    for sentence in sentences:
        # Convert to lowercase and split by whitespace
        words = sentence.lower().split()
        words_in_sentences.append(words)
    
    # Create vocabulary if not provided
    if vocab is None:
        # flatten the list of words from all sentences
        all_words = [word for words in words_in_sentences for word in words]
        # unique words, sort for consistency
        vocab = sorted(set(all_words))
    
    # Create a list of features (the vocabulary words)
    features = vocab
    
    # Initialize the embedding matrix with zeros
    embeddings = np.zeros((len(sentences), len(features)), dtype=np.int32)
    
    # Fill the embedding matrix
    for i, words in enumerate(words_in_sentences):
        # count occurrences of each word in my sentence
        word_counts = Counter(words)
        
        # update the embedding vector for my sentnece
        for j, word in enumerate(features):
            if word in word_counts:
                embeddings[i, j] = word_counts[word]
    
    return embeddings, features