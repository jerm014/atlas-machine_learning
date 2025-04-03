#!/usr/bin/env python3
"""n-gram BLEU"""
import math
from collections import Counter


def ngram_bleu(references, sentence, n):
    """ calculates n-gram BLEU score for a candidate sentence """

    # count n-grams in candidate sentence
    sentence_ngrams = ngrams(sentence, n)
    counts = Counter(sentence_ngrams)

    # count maximum reference counts for each n-gram
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = ngrams(ref, n)
        ref_counts = Counter(ref_ngrams)
        for ngram in ref_counts:
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0),
                                        ref_counts[ngram])

    # clipped count
    clipped_count = 0
    for ngram in counts:
        clipped_count += min(counts[ngram],
                             max_ref_counts.get(ngram, 0))

    # precision
    total_count = len(sentence_ngrams)
    precision = clipped_count / total_count if total_count > 0 else 0

    # calculate brevity penalty
    c = len(sentence)
    r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]
    if c > r:
        bp = 1
    elif c == 0:
        bp = 0
    else:
        bp = math.exp(1 - r / c)

    # BLEU score
    bleu = bp * precision
    return bleu


def ngrams(seq, n):
    """ tiny helper function to extract n-grams from a sequence """
    ngram_list = []
    for i in range(len(seq) - n + 1):
        ngram = tuple(seq[i:i + n])
        ngram_list.append(ngram)
    return ngram_list
