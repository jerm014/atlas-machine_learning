#!/usr/bin/env python3
"""cumulative n-gram BLEU"""
import math
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """ calculates cumulative n-gram BLEU score for a candidate sentence """
    precisions = []

    for i in range(1, n + 1):
        sentence_ngrams = ngrams(sentence, i)
        counts = Counter(sentence_ngrams)

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = ngrams(ref, i)
            ref_counts = Counter(ref_ngrams)
            for ngram in ref_counts:
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0),
                                            ref_counts[ngram])

        clipped_count = 0
        for ngram in counts:
            clipped_count += min(counts[ngram],
                                 max_ref_counts.get(ngram, 0))

        total_count = len(sentence_ngrams)
        precision = clipped_count / total_count if total_count > 0 else 0
        precisions.append(precision)

    # geometric mean of precisions in log space
    smooth_precisions = [p if p > 0 else 1e-9 for p in precisions]
    log_precisins = [math.log(p) for p in smooth_precisions]
    avg_log_precision = sum(log_precisins) / n
    geo_mean = math.exp(avg_log_precision)

    # calculate bp (brevity penalty)
    # c is the length of the candidate sentence (number of words)
    # r is the length of the reference closest in length to the candidate
    # bp penalizes short candidate sentences that might have high precision
    c = len(sentence)
    r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]
    if c > r:
        bp = 1
    elif c == 0:
        bp = 0
    else:
        bp = math.exp(1 - r / c)

    bleu = bp * geo_mean
    return bleu


def ngrams(seq, n):
    """ tiny helper function to extract n-grams from a sequence """
    ngram_list = []
    for i in range(len(seq) - n + 1):
        ngram = tuple(seq[i:i + n])
        ngram_list.append(ngram)
    return ngram_list
